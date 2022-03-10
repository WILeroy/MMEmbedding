import math


def warm_up_with_cosine(epoch, warm_up_epochs, total_epochs):
    if epoch <= warm_up_epochs:
        return (epoch + 1) / (warm_up_epochs + 1)
    else:
        return 0.5 * (math.cos((epoch - warm_up_epochs) / (
            total_epochs - warm_up_epochs) * math.pi) + 1)
