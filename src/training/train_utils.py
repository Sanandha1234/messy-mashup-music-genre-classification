import torch
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    """Train a model for one epoch."""

    model.train()

    total_loss = 0.0
    predictions = []
    targets = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions.extend(
            outputs.argmax(dim=1).detach().cpu().numpy()
        )
        targets.extend(
            labels.detach().cpu().numpy()
        )

    avg_loss = total_loss / max(len(dataloader), 1)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return avg_loss, accuracy, macro_f1


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """Evaluate a model without updating its parameters."""

    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions.extend(
                outputs.argmax(dim=1).cpu().numpy()
            )
            targets.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / max(len(dataloader), 1)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return avg_loss, accuracy, macro_f1