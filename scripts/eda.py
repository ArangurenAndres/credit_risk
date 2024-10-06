import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Rename the target variable column for clarity
    df.rename(columns={'default.payment.next.month': 'credit_default'}, inplace=True)

    # Set a consistent style for Seaborn
    sns.set(style="whitegrid")

    # 1. Distribution of the target variable (credit default)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='credit_default', data=df, palette='coolwarm')
    plt.title('Distribution of Credit Default', fontsize=16)
    plt.xlabel('Credit Default (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('results/eda_credit_default_distribution.png', bbox_inches='tight')
    plt.show()

    # 2. Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.savefig('results/eda_correlation_heatmap.png', bbox_inches='tight')
    plt.show()

    # 3. Boxplots for key numerical features
    numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
    for feature in numerical_features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='credit_default', y=feature, data=df, palette='coolwarm')
        plt.title(f'{feature} by Credit Default', fontsize=16)
        plt.savefig(f'results/eda_boxplot_{feature}.png', bbox_inches='tight')
        plt.show()

    # 4. Histograms for key numerical features
    for feature in numerical_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], bins=30, kde=True, color='blue')
        plt.title(f'Distribution of {feature}', fontsize=16)
        plt.savefig(f'results/eda_histogram_{feature}.png', bbox_inches='tight')
        plt.show()

    # 5. Countplots for categorical features (PAY_0, PAY_2, etc.)
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    for feature in categorical_features:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=feature, hue='credit_default', data=df, palette='coolwarm')
        plt.title(f'{feature} by Credit Default', fontsize=16)
        plt.savefig(f'results/eda_countplot_{feature}.png', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Perform EDA on the dataset
    perform_eda("data/credit_card_default.csv")
