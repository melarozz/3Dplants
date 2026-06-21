import numpy as np
import matplotlib.pyplot as plt


def save_ppfd_histogram(ppfd: np.ndarray, path: str):
    plt.figure()
    plt.hist(ppfd, bins=50, color="green", alpha=0.7)
    plt.xlabel("PPFD (µmol/m²/s)")
    plt.ylabel("Count")
    plt.title("PPFD Distribution")
    plt.savefig(path)
    plt.close()


def save_colorbar(path: str, max_ppfd: float = 1000):
    gradient = np.linspace(0, max_ppfd, 256)
    gradient = gradient[:, None]

    plt.figure(figsize=(2, 6))
    plt.imshow(gradient, cmap="plasma", aspect="auto")
    plt.gca().set_yticks(np.linspace(0, 255, 5))
    plt.gca().set_yticklabels(np.linspace(0, max_ppfd, 5).astype(int))
    plt.xticks([])
    plt.ylabel("PPFD (µmol/m²/s)")
    plt.title("Scale")
    plt.savefig(path)
    plt.close()