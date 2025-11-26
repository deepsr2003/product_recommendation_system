import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_visualizations():
    """
    Create comprehensive visualizations for the RFM analysis
    """
    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Load data
    rfm_data = pd.read_csv("data/rfm_features.csv")
    feature_importance = pd.read_csv("outputs/feature_importance.csv")
    test_results = pd.read_csv("outputs/test_results.csv")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Product Recommendation System - RFM Analysis", fontsize=16, fontweight="bold"
    )

    # 1. RFM Score Distribution
    ax1 = axes[0, 0]
    rfm_scores = rfm_data[["recency_score", "frequency_score", "monetary_score"]]
    for i, score in enumerate(["recency_score", "frequency_score", "monetary_score"]):
        ax1.hist(
            rfm_data[score], alpha=0.6, label=score.replace("_", " ").title(), bins=5
        )
    ax1.set_title("RFM Score Distribution")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Feature Importance
    ax2 = axes[0, 1]
    sns.barplot(data=feature_importance, x="importance", y="feature", ax=ax2)
    ax2.set_title("Feature Importance\n(CatBoost Model)")
    ax2.set_xlabel("Importance (%)")
    ax2.set_ylabel("Features")

    # Add percentage labels
    for i, (importance, feature) in enumerate(
        zip(feature_importance["importance"], feature_importance["feature"])
    ):
        ax2.text(importance + 0.5, i, f"{importance:.1f}%", va="center")

    # 3. Customer Segments
    ax3 = axes[0, 2]
    segment_counts = rfm_data["segment_category"].value_counts()
    colors = sns.color_palette("husl", len(segment_counts))
    ax3.pie(
        segment_counts.values,
        labels=segment_counts.index,
        autopct="%1.1f%%",
        colors=colors,
    )
    ax3.set_title("Customer Segments Distribution")

    # 4. Monetary vs Recency Scatter
    ax4 = axes[1, 0]
    scatter = ax4.scatter(
        rfm_data["recency"],
        rfm_data["monetary"],
        c=rfm_data["is_high_value"],
        alpha=0.6,
        cmap="RdYlBu",
    )
    ax4.set_title("Monetary Value vs Recency")
    ax4.set_xlabel("Recency (days ago)")
    ax4.set_ylabel("Monetary Value ($)")
    plt.colorbar(scatter, ax=ax4, label="High Value Customer")

    # 5. RFM Heatmap
    ax5 = axes[1, 1]
    rfm_matrix = (
        rfm_data.groupby(["recency_score", "frequency_score"])["monetary"]
        .mean()
        .unstack()
    )
    sns.heatmap(rfm_matrix, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax5)
    ax5.set_title("Average Monetary Value\nby Recency & Frequency Scores")
    ax5.set_xlabel("Frequency Score")
    ax5.set_ylabel("Recency Score")

    # 6. Model Confusion Matrix
    ax6 = axes[1, 2]
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_results["actual"], test_results["predicted"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax6,
        xticklabels=["Low Value", "High Value"],
        yticklabels=["Low Value", "High Value"],
    )
    ax6.set_title("Model Confusion Matrix\n(Test Set: 868 samples)")
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("outputs/rfm_analysis_dashboard.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create individual feature analysis plots
    create_feature_analysis_plots(rfm_data)


def create_feature_analysis_plots(rfm_data):
    """
    Create detailed feature analysis plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "RFM Feature Analysis by Customer Value", fontsize=14, fontweight="bold"
    )

    # Recency analysis
    ax1 = axes[0]
    sns.boxplot(data=rfm_data, x="is_high_value", y="recency", ax=ax1)
    ax1.set_title("Recency Distribution")
    ax1.set_xlabel("High Value Customer")
    ax1.set_ylabel("Recency (days ago)")
    ax1.set_xticklabels(["No", "Yes"])

    # Frequency analysis
    ax2 = axes[1]
    sns.boxplot(data=rfm_data, x="is_high_value", y="frequency", ax=ax2)
    ax2.set_title("Frequency Distribution")
    ax2.set_xlabel("High Value Customer")
    ax2.set_ylabel("Frequency (transactions)")
    ax2.set_xticklabels(["No", "Yes"])

    # Monetary analysis
    ax3 = axes[2]
    sns.boxplot(data=rfm_data, x="is_high_value", y="monetary", ax=ax3)
    ax3.set_title("Monetary Distribution")
    ax3.set_xlabel("High Value Customer")
    ax3.set_ylabel("Monetary Value ($)")
    ax3.set_xticklabels(["No", "Yes"])

    plt.tight_layout()
    plt.savefig("outputs/feature_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_visualizations()
    print("Visualizations saved to outputs/")
    print("- rfm_analysis_dashboard.png")
    print("- feature_analysis.png")
