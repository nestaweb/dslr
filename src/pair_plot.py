from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import DSLR


def _load_dataset() -> pd.DataFrame:
    """
    Load the training dataset if available; otherwise, build a small synthetic
    example with a 'Hogwarts House' column and a few numeric features.
    """
    try:
        df = pd.read_csv(DSLR.trainfilepath)
        # Check si le split s'est bien effectué
        if 'Hogwarts House' in df.columns:
            return df
    except FileNotFoundError:
        pass


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Select up to max_features numeric columns using existing helper; if none,
    infer from dataframe directly.
    """
    features = DSLR.getNumericalFeature()
    features = [f for f in features if f in df.columns]
    if not features:
        numeric_df = df.select_dtypes(include=[np.number])
        features = list(numeric_df.columns)
    return features


def _pair_grid(df: pd.DataFrame, features: List[str], hue_col: str = 'Hogwarts House') -> None:
	# Pour fit le pair plot dans le screen
    def _get_screen_size_px():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    screen_w_px, screen_h_px = _get_screen_size_px()
    dpi = float(plt.rcParams.get('figure.dpi', 100))

    houses = [h for h in df[hue_col].dropna().unique()]
    houses.sort()

    n = len(features)
    base = 3.2
    desired_w_in = base * n
    desired_h_in = base * n

    margin_in = 1.0
    max_w_in = max((screen_w_px / dpi) - margin_in, 2.0)
    max_h_in = max((screen_h_px / dpi) - margin_in, 2.0)
    scale = min(1.0, max_w_in / desired_w_in, max_h_in / desired_h_in)
    figsize = (desired_w_in * scale, desired_h_in * scale)

    fig, axes = plt.subplots(n, n, figsize=figsize)

	# si qu'une seule feature -> qu'un seul axe
    if n == 1:
        axes = np.array([[axes]])

    colors = {
        'Ravenclaw': '#1f77b4',
        'Slytherin': '#2ca02c',
        'Gryffindor': '#d62728',
        'Hufflepuff': '#ffbf00',
    }

    for i, y_feature in enumerate(features):
        for j, x_feature in enumerate(features):
            ax = axes[i, j]

            if i == j:
                # si feature x == feature y --> histogram
                for house in houses:
                    vals = df.loc[df['Hogwarts House'] == house, y_feature].dropna()
                    if len(vals) > 0:
                        ax.hist(vals, bins=20, alpha=0.5, color=colors.get(house, None))
                ax.set_ylabel('Count')
            else:
                # si feature x != feature y --> scatter plot
                for house in houses:
                    sub = df.loc[df['Hogwarts House'] == house, [x_feature, y_feature]].dropna()
                    if len(sub) > 0:
                        ax.scatter(
                            sub[x_feature],
                            sub[y_feature],
                            s=10,
                            alpha=0.6,
                            color=colors.get(house, None),
                            label=house if (i == 0 and j == 1) else None,
                        )

            if i == n - 1:
                ax.set_xlabel(x_feature)
            else:
                ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel(y_feature)
            else:
                ax.set_ylabel('')

    # Legend
    handles, labels = axes[0, 1].get_legend_handles_labels() if n > 1 else axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False)

    fig.suptitle('Pair Plot', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def main():
    df = _load_dataset()
    if 'Hogwarts House' not in df.columns:
        raise RuntimeError("Dataset must include 'Hogwarts House' column")
    features = _select_features(df)
    if not features:
        raise RuntimeError('No numeric features available to plot')
    _pair_grid(df, features, 'Hogwarts House')


if __name__ == '__main__':
    main()


