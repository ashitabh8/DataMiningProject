import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_validation_acc = 0.0
    training_acc_for_best_validation = 0.0
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            label_class = torch.argmax(labels, dim=1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # breakpoint()
                # label_class = torch.argmax(labels)[1]
                # label_class = torch.argmax(labels, dim=1)
                loss = criterion(outputs, label_class)

                # Backward + optimize
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            # breakpoint()
            running_corrects += torch.sum(preds == label_class.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} - Training loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            label_class = torch.argmax(labels, dim=1)

            # Forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Statistics
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == label_class.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        if val_acc > best_validation_acc:
            best_validation_acc = val_acc
            training_acc_for_best_validation = epoch_acc
            # best_model_wts = model.state_dict()

        print(f'Epoch {epoch}/{num_epochs - 1} - Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

    print(f"Best validation accuracy: {best_validation_acc:.4f}, Training accuracy for best validation: {training_acc_for_best_validation:.4f}")
    print('Training complete')
    return model

