import matplotlib.pyplot as plt

# Evaluate global model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=32)

accuracy = evaluate_model(global_model, test_loader)
print(f"Global Model Accuracy: {accuracy}%")

# Plot results
plt.plot(range(1, epochs + 1), accuracy, label='Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
