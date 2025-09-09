import torch
import torchhd as hd
import torch.nn as nn
from torchhd.models import Centroid
from args import args
from data_loader import get_dataset, get_dataloader_HV
from hd_transformer import HDTransformerModel


def train(model, train_loader, optimizer, device, args):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in train_loader:
        optimizer.zero_grad()
        data, label = batch

        label = label[:, 0].long().to(device[0])
        data = data.to(device[0])

        outputs, _, _ = model(data)
        loss = nn.CrossEntropyLoss()(outputs, label)

        loss.backward()
        if args.maxL2 is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.maxL2)  # 梯度裁剪, nax_norm规定最大的L2范数
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == label).sum().item()
        total_preds += label.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds

    return epoch_loss, epoch_acc


def validate(model, val_loader, device, args):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    correct_preds_hd = 0
    total_preds = 0

    with torch.no_grad():
        for batch in val_loader:
            data, label = batch

            label = label[:, 0].long().to(device[0])
            data = data.to(device[0])

            outputs, query_vectors, associate_memory = model(data)
            loss = nn.CrossEntropyLoss()(outputs, label)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == label).sum().item()

            similarity_metrics = hd.hamming_similarity(query_vectors, associate_memory)
            _, predicted_hd = torch.max(similarity_metrics, 1)
            correct_preds_hd += (predicted_hd == label).sum().item()

            total_preds += label.size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct_preds / total_preds
        epoch_acc_hd = correct_preds_hd / total_preds

    return epoch_loss, epoch_acc, epoch_acc_hd


def main():
    if args.multigpu is None:
        device = torch.device("cpu")
    elif len(args.multigpu) == 1:
        device = [torch.device(f'cuda:{args.multigpu[0]}')]
    else:
        device = [torch.device(f'cuda:{gpu_id}') for gpu_id in args.multigpu]

    print(f"Using device(s): {device}")

    train_dataset, val_dataset, test_dataset, class_number, input_dim = get_dataset(args)
    train_loader, val_loader, test_loader = get_dataloader_HV(train_dataset, val_dataset, test_dataset, args)
    print("Data preprocess complete")

    if len(args.multigpu) > 1:
        model = nn.DataParallel(HDTransformerModel(input_dim=args.HV_dim, num_classes=class_number,
                                                   seq_len=train_dataset.data.max_seq_len, dropout=args.dropout),
                                device_ids=device)
    else:
        model = HDTransformerModel(input_dim=args.HV_dim, num_classes=class_number,
                                   seq_len=train_dataset.data.max_seq_len, dropout=args.dropout)
    model = model.to(device[0])

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)

    num_epochs = args.epoch
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device, args)
        val_loss, val_acc, _ = validate(model, val_loader, device, args)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    print(args)
    main()
