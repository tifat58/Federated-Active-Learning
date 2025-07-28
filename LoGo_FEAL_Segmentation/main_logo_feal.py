import argparse
import yaml
import torch
import numpy as np
import random
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

from data.fedpolyp_loader import get_fedpolyp_loaders, load_txt_splits as load_csv_splits
from models.evidential_unet import EvidentialUNet
from trainers.local_update import LocalUpdate
from trainers.global_aggregator import aggregate_models
from query_strategies.evidential_uncertainty import select_uncertain_samples
from utils.clustering import get_client_clusters
from utils.metrics import evaluate

CHECKPOINT_PATH = "checkpoints/logo_feal_checkpoint.pt"

# Create unique log file using current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"logs/al_fl_log_{timestamp}.txt"
os.makedirs("logs", exist_ok=True)
log_file = open(log_path, "w")

def log_config(config):
    log_file.write("[CONFIGURATION USED IN THIS RUN]\n")
    for section, params in config.items():
        log_file.write(f"{section}:\n")
        if isinstance(params, dict):
            for key, value in params.items():
                log_file.write(f"  {key}: {value}\n")
        else:
            log_file.write(f"  {params}\n")
    log_file.write("\n[BEGINNING LOGGING OF ROUNDS...]\n\n")
    log_file.flush()

def log_status(al_round, fl_round, dice_score, labeled_data):
    log_file.write(f"\n[LOG] AL round {al_round}, FL round {fl_round}\n")
    log_file.write(f"[LOG] Global Mean Dice Score: {dice_score:.4f}\n")
    log_file.write("[LOG] Labeled training size per client:\n")
    for cid in sorted(labeled_data):
        log_file.write(f"Client {cid}: {len(labeled_data[cid])} samples\n")
    log_file.flush()

def save_checkpoint(round_idx, global_model, client_models, labeled_data, unlabeled_data, dice_score):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'round_idx': round_idx,
        'global_model_state': global_model.state_dict(),
        'client_model_states': {cid: m.state_dict() for cid, m in client_models.items()},
        'labeled_data': labeled_data,
        'unlabeled_data': unlabeled_data,
        'dice_score': dice_score
    }, CHECKPOINT_PATH)

def load_checkpoint(global_model, client_models):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        global_model.load_state_dict(checkpoint['global_model_state'])
        for cid in client_models:
            client_models[cid].load_state_dict(checkpoint['client_model_states'][cid])
        return (
            checkpoint['round_idx'],
            checkpoint['labeled_data'],
            checkpoint['unlabeled_data'],
            checkpoint['dice_score']
        )
    else:
        return 0, None, None, 0.0

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fedpolyp_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    log_config(config)

    apply_cosine_filter = config.get('sampling', {}).get('use_cosine_filter', False)
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_splits = load_csv_splits(config['dataset']['csv_dir'])
    train_dataset, user_groups, test_loader = get_fedpolyp_loaders(config, csv_splits)

    global_model = EvidentialUNet(config['model']['num_classes']).to(device)
    client_models = {cid: EvidentialUNet(config['model']['num_classes']).to(device) for cid in user_groups.keys()}

    labeled_data, unlabeled_data = {}, {}
    for cid in user_groups:
        all_indices = list(user_groups[cid])
        min_labeled = max(1, int(len(all_indices) * config['initial_label_rate']))
        print(f"[Client {cid}] Total samples: {len(all_indices)}")
        if min_labeled >= len(all_indices):
            min_labeled = len(all_indices) - 1  # keep at least 1 for unlabeled
        if min_labeled <= 0:
            min_labeled = 1  # avoid zero labeled
        labeled_idx, unlabeled_idx = train_test_split(
            all_indices,
            train_size=min_labeled,
            random_state=config['seed']
        )
        labeled_data[cid] = labeled_idx
        unlabeled_data[cid] = unlabeled_idx

    start_round, checkpoint_labeled, checkpoint_unlabeled, best_dice = load_checkpoint(global_model, client_models)
    if checkpoint_labeled is not None:
        labeled_data = checkpoint_labeled
        unlabeled_data = checkpoint_unlabeled
        print(f"Resuming from round {start_round + 1}...")

    for round in range(start_round, config['total_rounds']):
        print(f"\n--- Federated Round {round + 1} ---")

        local_weights, local_uncertainties = {}, {}
        for cid in user_groups:
            local = LocalUpdate(
                model=client_models[cid],
                dataset=train_dataset,
                idxs=labeled_data[cid],
                device=device,
                lr=config['learning_rate'],
                local_epochs=config['local_epochs'],
                batch_size=config['batch_size'],
                kl_weight=config['kl_weight'],
                annealing_step=config['annealing_step'],
                num_classes=config['model']['num_classes'],
                cid=cid
            )
            updated_model_state, uncertainty_score = local.train(task='segmentation', lam=config['evidential_loss']['lambda'])
            local_weights[cid] = updated_model_state
            local_uncertainties[cid] = uncertainty_score

        global_weights = aggregate_models(list(local_weights.values()), list(local_uncertainties.values()), config['uncertainty']['beta'])
        global_model.load_state_dict(global_weights)
        for cid in client_models:
            client_models[cid].load_state_dict(global_model.state_dict())

        if (round + 1) % config['al_period'] == 0 and (round + 1) != config['total_rounds']:
            print(f"\n--- Active Learning Round {(round // config['al_period']) + 1} ---")
            
            client_ids = list(user_groups.keys())  # e.g., [1, 2, 3, 4]
            client_loaders = {
                cid: torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_dataset, labeled_data[cid]),
                    batch_size=config['batch_size'],
                    shuffle=False
                ) for cid in client_ids
            }

            # Pass ordered loaders list to match model order
            cluster_groups = get_client_clusters(client_models, client_loaders, device)

            # Convert cluster indices to actual client IDs
            mapped_clusters = []
            for group in cluster_groups:
                mapped_group = [client_ids[idx] for idx in group]
                mapped_clusters.append(mapped_group)
            cluster_groups = mapped_clusters

            max_clients = config['num_clients']
            selected_clients = [random.choice(group) for group in cluster_groups if group]
            remaining_candidates = set(cid for group in cluster_groups for cid in group) - set(selected_clients)
            needed = max_clients - len(selected_clients)
            if needed > 0:
                selected_clients += random.sample(list(remaining_candidates), min(needed, len(remaining_candidates)))
            
            print(f"Selected client(s): {selected_clients}")

            for cid in selected_clients:
                unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_data[cid])
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=config['batch_size'], shuffle=False)

                al_round_id = (round + 1) // config['al_period']
                if 'label_budget_schedule' in config:
                    budget_schedule = config['label_budget_schedule']
                    query_budget = budget_schedule[min(al_round_id - 1, len(budget_schedule) - 1)]
                else:
                    query_budget = config['budget_per_round']

                selected_indices = select_uncertain_samples(
                    model=client_models[cid],
                    unlabeled_loader=unlabeled_loader,
                    device=device,
                    query_budget=query_budget,
                    apply_cosine_filter=apply_cosine_filter
                )
                selected_indices = [int(i) for i in selected_indices]
                labeled_data[cid].extend(selected_indices)
                unlabeled_data[cid] = [idx for idx in unlabeled_data[cid] if idx not in selected_indices]

            print("\n[Info] Updated labeled training sizes per client:")
            for cid in labeled_data:
                print(f"Client {cid}: {len(labeled_data[cid])} samples")

        dice_score = evaluate(global_model, test_loader, device, task='segmentation')
        if dice_score > best_dice:
            best_dice = dice_score

        print(f"[Round {round+1}] Mean Dice Score: {dice_score:.4f}")
        al_round_id = (round + 1) // config['al_period'] if (round + 1) % config['al_period'] == 0 else "-"
        log_status(al_round_id, round + 1, dice_score, labeled_data)
        save_checkpoint(round + 1, global_model, client_models, labeled_data, unlabeled_data, dice_score)

    total_minutes = (time.time() - start_time) / 60
    log_file.write(f"\n[LOG] Total training time: {total_minutes:.2f} minutes\n")
    print(f"\nTotal training time: {total_minutes:.2f} minutes")
    log_file.close()

if __name__ == '__main__':
    main()