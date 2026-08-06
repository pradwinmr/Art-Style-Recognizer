import matplotlib
matplotlib.use('Agg') # Bypassing the Windows Tkinter error!
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    print("Generating Exploratory Data Analysis (EDA) graph...")
    
    # Using the exact counts we discovered during the recursive scan
    data = {
        'Art Style': [
            'Academic Art', 'Art Nouveau', 'Baroque', 'Expressionism', 
            'Japanese Art', 'Neoclassicism', 'Primitivism', 'Realism', 
            'Renaissance', 'Rococo', 'Romanticism', 'Symbolism', 'Western Medieval'
        ],
        'Image Count': [
            1305, 3035, 5312, 2607, 2235, 3115, 1324, 5373, 
            6192, 2521, 6813, 1510, 1158
        ]
    }
    
    # Convert to a DataFrame and sort it from highest to lowest for better readability
    df = pd.DataFrame(data)
    df = df.sort_values(by='Image Count', ascending=False)
    
    # Draw the bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Image Count', y='Art Style', data=df, palette='magma')
    
    plt.title('Dataset Imbalance: Image Counts per Art Style', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Images', fontsize=12, fontweight='bold')
    plt.ylabel('Art Style', fontsize=12, fontweight='bold')
    
    # Add the exact numbers to the end of each bar
    for index, value in enumerate(df['Image Count']):
        plt.text(value + 50, index, str(value), va='center', fontsize=10)
        
    plt.tight_layout()
    
    # Save the visual graph as an image
    plt.savefig('eda_distribution.png', dpi=300)
    print("Success! Saved EDA graph as 'eda_distribution.png' in your project folder!")

if __name__ == '__main__':
    main()