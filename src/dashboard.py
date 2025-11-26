import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class RFMDashboard:
    def __init__(self):
        # Load model and data
        self.model, self.feature_columns = (
            joblib.load("models/catboost_model.pkl"),
            joblib.load("models/feature_columns.pkl"),
        )
        self.rfm_data = pd.read_csv("data/rfm_features.csv")

        # Get median monetary value for classification
        self.median_monetary = self.rfm_data["monetary"].median()

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        """Create interactive widgets for RFM simulation"""
        # Title
        title = HTML(
            "<h2>Product Recommendation System - RFM Simulation Dashboard</h2>"
        )

        # RFM Score sliders
        self.recency_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Recency Score:",
            style={"description_width": "initial"},
            continuous_update=False,
        )

        self.frequency_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Frequency Score:",
            style={"description_width": "initial"},
            continuous_update=False,
        )

        self.monetary_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Monetary Score:",
            style={"description_width": "initial"},
            continuous_update=False,
        )

        # Raw value inputs
        self.recency_days = widgets.IntText(
            value=30,
            min=1,
            max=365,
            description="Recency (days):",
            style={"description_width": "initial"},
        )

        self.frequency_count = widgets.IntText(
            value=5,
            min=1,
            max=100,
            description="Frequency (count):",
            style={"description_width": "initial"},
        )

        self.monetary_value = widgets.FloatText(
            value=100.0,
            min=10.0,
            max=10000.0,
            description="Monetary ($):",
            style={"description_width": "initial"},
        )

        # Prediction button
        self.predict_button = widgets.Button(
            description="Predict Recommendation", button_style="success", icon="check"
        )

        # Output area
        self.output_area = widgets.Output()

        # Arrange widgets
        input_widgets = widgets.VBox(
            [
                HTML("<h3>RFM Scores (1-5 scale)</h3>"),
                self.recency_slider,
                self.frequency_slider,
                self.monetary_slider,
                HTML("<hr>"),
                HTML("<h3>Raw RFM Values</h3>"),
                self.recency_days,
                self.frequency_count,
                self.monetary_value,
                HTML("<hr>"),
                self.predict_button,
            ]
        )

        # Main layout
        self.main_layout = widgets.HBox([input_widgets, self.output_area])

        # Connect event handlers
        self.predict_button.on_click(self.make_prediction)
        self.recency_slider.observe(self.update_raw_values, names="value")
        self.frequency_slider.observe(self.update_raw_values, names="value")
        self.monetary_slider.observe(self.update_raw_values, names="value")

        # Display
        display(title, self.main_layout)

    def update_raw_values(self, change):
        """Update raw values based on score changes"""
        # Map scores to approximate raw values
        recency_mapping = {1: 180, 2: 90, 3: 30, 4: 14, 5: 7}
        frequency_mapping = {1: 1, 2: 3, 3: 5, 4: 10, 5: 20}
        monetary_mapping = {1: 50, 2: 100, 3: 200, 4: 500, 5: 1000}

        self.recency_days.value = recency_mapping[self.recency_slider.value]
        self.frequency_count.value = frequency_mapping[self.frequency_slider.value]
        self.monetary_value.value = monetary_mapping[self.monetary_slider.value]

    def calculate_rfm_scores(self, recency, frequency, monetary):
        """Calculate RFM scores from raw values"""
        # Simple scoring based on percentiles from our data
        recency_score = (
            5
            if recency <= 7
            else (
                4
                if recency <= 14
                else (3 if recency <= 30 else (2 if recency <= 90 else 1))
            )
        )
        frequency_score = (
            5
            if frequency >= 20
            else (
                4
                if frequency >= 10
                else (3 if frequency >= 5 else (2 if frequency >= 3 else 1))
            )
        )
        monetary_score = (
            5
            if monetary >= 1000
            else (
                4
                if monetary >= 500
                else (3 if monetary >= 200 else (2 if monetary >= 100 else 1))
            )
        )

        return recency_score, frequency_score, monetary_score

    def make_prediction(self, button):
        """Make prediction using the trained model"""
        with self.output_area:
            clear_output()

            # Get input values
            recency = self.recency_days.value
            frequency = self.frequency_count.value
            monetary = self.monetary_value.value

            # Calculate scores
            recency_score, frequency_score, monetary_score = self.calculate_rfm_scores(
                recency, frequency, monetary
            )

            # Create feature array
            features = np.array(
                [
                    [
                        recency,
                        frequency,
                        monetary,
                        recency_score,
                        frequency_score,
                        monetary_score,
                    ]
                ]
            )
            feature_df = pd.DataFrame(features, columns=self.feature_columns)

            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            prediction_proba = self.model.predict_proba(feature_df)[0]

            # Determine recommendation
            recommendation = "RECOMMEND" if prediction == 1 else "DO NOT RECOMMEND"
            confidence = (
                prediction_proba[1] * 100
                if prediction == 1
                else prediction_proba[0] * 100
            )

            # Display results
            self.display_prediction_results(
                recency,
                frequency,
                monetary,
                recency_score,
                frequency_score,
                monetary_score,
                recommendation,
                confidence,
                prediction_proba,
            )

    def display_prediction_results(
        self,
        recency,
        frequency,
        monetary,
        recency_score,
        frequency_score,
        monetary_score,
        recommendation,
        confidence,
        prediction_proba,
    ):
        """Display prediction results with visualizations"""

        # Results card
        result_color = "green" if recommendation == "RECOMMEND" else "red"
        result_html = f"""
        <div style="border: 2px solid {result_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: {result_color}; text-align: center;">{recommendation}</h3>
            <p style="text-align: center;"><strong>Confidence: {confidence:.1f}%</strong></p>
            <p style="text-align: center;">High Value Probability: {prediction_proba[1] * 100:.1f}%</p>
        </div>
        """
        display(HTML(result_html))

        # RFM Analysis
        analysis_html = f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h4>RFM Analysis:</h4>
            <ul>
                <li><strong>Recency:</strong> {recency} days ago (Score: {recency_score}/5)</li>
                <li><strong>Frequency:</strong> {frequency} transactions (Score: {frequency_score}/5)</li>
                <li><strong>Monetary:</strong> ${monetary:.2f} (Score: {monetary_score}/5)</li>
                <li><strong>RFM Segment:</strong> {recency_score}{frequency_score}{monetary_score}</li>
                <li><strong>vs Median Monetary:</strong> {"Above" if monetary >= self.median_monetary else "Below"} median (${self.median_monetary:.2f})</li>
            </ul>
        </div>
        """
        display(HTML(analysis_html))

        # Create visualization
        self.create_customer_profile_plot(recency, frequency, monetary, prediction)

    def create_customer_profile_plot(self, recency, frequency, monetary, prediction):
        """Create a plot showing customer profile relative to dataset"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Customer position on RFM map
        high_value_mask = self.rfm_data["is_high_value"] == 1
        low_value_mask = self.rfm_data["is_high_value"] == 0

        ax1.scatter(
            self.rfm_data[low_value_mask]["recency"],
            self.rfm_data[low_value_mask]["monetary"],
            alpha=0.3,
            c="blue",
            label="Low Value",
            s=20,
        )
        ax1.scatter(
            self.rfm_data[high_value_mask]["recency"],
            self.rfm_data[high_value_mask]["monetary"],
            alpha=0.3,
            c="red",
            label="High Value",
            s=20,
        )

        # Highlight current customer
        customer_color = "red" if prediction == 1 else "blue"
        ax1.scatter(
            recency,
            monetary,
            c=customer_color,
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Current Customer",
            zorder=5,
        )

        ax1.set_xlabel("Recency (days ago)")
        ax1.set_ylabel("Monetary Value ($)")
        ax1.set_title("Customer Profile - Recency vs Monetary")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Feature comparison
        features = ["Recency", "Frequency", "Monetary"]
        customer_values = [recency, frequency, monetary]
        dataset_means = [
            self.rfm_data["recency"].mean(),
            self.rfm_data["frequency"].mean(),
            self.rfm_data["monetary"].mean(),
        ]

        x = np.arange(len(features))
        width = 0.35

        ax2.bar(
            x - width / 2,
            customer_values,
            width,
            label="Current Customer",
            color=customer_color,
        )
        ax2.bar(x + width / 2, dataset_means, width, label="Dataset Average", alpha=0.7)

        ax2.set_xlabel("Features")
        ax2.set_ylabel("Values")
        ax2.set_title("Feature Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(features)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def run_dashboard():
    """Run the interactive dashboard"""
    dashboard = RFMDashboard()


if __name__ == "__main__":
    run_dashboard()
