import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# Create img directory if it doesn't exist
os.makedirs('img', exist_ok=True)

class EMNISTClusteringAnalysis:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_scaled = None
        self.results = {}
        
    def load_data(self, n_samples=10000):
        """Load and preprocess MNIST dataset"""
        print("Loading EMNIST MNIST dataset...")
        
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # Sample data
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(mnist.data), n_samples, replace=False)
        self.X = mnist.data[indices]
        self.y = mnist.target[indices].astype(int)
        
        # Normalize pixel values
        self.X = self.X / 255.0
        
        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Digit distribution: {Counter(self.y)}")
        
    def perform_kmeans_clustering(self, n_clusters, n_trials=10):
        """Perform K-means clustering with multiple trials"""
        print(f"\nPerforming k-means clustering for {n_clusters} clusters...")
        
        best_inertia = float('inf')
        best_kmeans = None
        best_labels = None
        
        for trial in range(n_trials):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                init='k-means++',  # Improved centroid initialization
                n_init=1,
                max_iter=300,
                random_state=trial,
                algorithm='lloyd'
            )
            
            labels = kmeans.fit_predict(self.X_scaled)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
                best_labels = labels
                
        print(f"Best inertia after {n_trials} trials: {best_inertia:.2f}")
        
        return best_kmeans, best_labels, best_inertia
    
    def create_assignment_matrix(self, labels, n_clusters):
        """Create assignment matrix showing percentage of each digit in each cluster"""
        assignment_matrix = np.zeros((10, n_clusters))
        
        for digit in range(10):
            digit_indices = np.where(self.y == digit)[0]
            if len(digit_indices) > 0:
                digit_labels = labels[digit_indices]
                
                for cluster in range(n_clusters):
                    count = np.sum(digit_labels == cluster)
                    percentage = (count / len(digit_indices)) * 100
                    assignment_matrix[digit, cluster] = percentage
                
        return assignment_matrix
    
    def plot_assignment_matrix(self, assignment_matrix, n_clusters, title_suffix=""):
        """Plot assignment matrix as heatmap"""
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(
            assignment_matrix,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
            yticklabels=[f'Digit {i}' for i in range(10)],
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title(f'K-means: Digit Assignment to Clusters{title_suffix}\n({n_clusters} clusters)')
        plt.xlabel('Clusters')
        plt.ylabel('Digits')
        plt.tight_layout()
        
        # Save the figure
        filename = f"img/kmeans_{n_clusters}_assignment_matrix.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_centroids(self, kmeans, n_clusters, title_suffix=""):
        """Visualize cluster centroids as images"""
        centroids = kmeans.cluster_centers_
        
        # Determine grid size based on number of clusters
        if n_clusters == 10:
            grid_size = (2, 5)
        elif n_clusters == 15:
            grid_size = (3, 5)
        elif n_clusters == 20:
            grid_size = (4, 5)
        elif n_clusters == 30:
            grid_size = (5, 6)
        else:
            grid_size = (int(np.ceil(np.sqrt(n_clusters))), int(np.ceil(np.sqrt(n_clusters))))
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        
        for i in range(n_clusters):
            # Reshape centroid back to 28x28 image
            centroid_image = centroids[i].reshape(28, 28)
            
            axes[i].imshow(centroid_image, cmap='gray')
            axes[i].set_title(f'Cluster {i}')
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'K-means: Cluster Centroids{title_suffix}\n({n_clusters} clusters)', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        filename = f"img/kmeans_{n_clusters}_centroids.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_cluster_purity(self, labels, n_clusters):
        """Analyze cluster purity and dominant digits"""
        cluster_analysis = {}
        
        print(f"\nCluster Analysis for {n_clusters} clusters:")
        print("-" * 50)
        
        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            cluster_digits = self.y[cluster_indices]
            
            # Count digits in this cluster
            digit_counts = Counter(cluster_digits)
            total_in_cluster = len(cluster_digits)
            
            if total_in_cluster > 0:
                # Find dominant digit
                dominant_digit = digit_counts.most_common(1)[0][0]
                dominant_percentage = (digit_counts[dominant_digit] / total_in_cluster) * 100
                
                cluster_analysis[cluster] = {
                    'dominant_digit': dominant_digit,
                    'dominant_percentage': dominant_percentage,
                    'total_samples': total_in_cluster,
                    'digit_distribution': dict(digit_counts)
                }
                
                print(f"Cluster {cluster:2d}: Dominant digit {dominant_digit} ({dominant_percentage:.1f}%), "
                      f"Total samples: {total_in_cluster}")
                
                # Show distribution of all digits in this cluster
                if len(digit_counts) > 1:
                    top_3 = digit_counts.most_common(3)
                    dist_str = ", ".join([f"{digit}:{count}" for digit, count in top_3])
                    print(f"           Distribution: {dist_str}")
            
        return cluster_analysis
    
    def suggest_cluster_merging(self, cluster_analysis):
        """Suggest which clusters could be merged for digit classification"""
        print(f"\nCluster Merging Suggestions:")
        print("-" * 40)
        
        # Group clusters by dominant digit
        digit_to_clusters = {}
        for cluster, info in cluster_analysis.items():
            dominant = info['dominant_digit']
            if dominant not in digit_to_clusters:
                digit_to_clusters[dominant] = []
            digit_to_clusters[dominant].append((cluster, info['dominant_percentage']))
        
        for digit in sorted(digit_to_clusters.keys()):
            clusters = digit_to_clusters[digit]
            if len(clusters) > 1:
                clusters.sort(key=lambda x: x[1], reverse=True)  # Sort by purity
                cluster_ids = [str(c[0]) for c in clusters]
                purities = [f"{c[1]:.1f}%" for c in clusters]
                print(f"Digit {digit}: Clusters {', '.join(cluster_ids)} "
                      f"(purities: {', '.join(purities)})")
                print(f"         → Consider merging these clusters")
    
    def run_complete_analysis(self):
        """Run complete clustering analysis for different numbers of clusters"""
        self.load_data(n_samples=15000)
        
        cluster_numbers = [10, 15, 20, 30]
        
        print(f"\n{'='*80}")
        print("K-MEANS CLUSTERING ANALYSIS ON MNIST DATASET")
        print(f"{'='*80}")
        
        for n_clusters in cluster_numbers:
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR {n_clusters} CLUSTERS")
            print(f"{'='*60}")
            
            # Perform clustering
            kmeans, labels, inertia = self.perform_kmeans_clustering(n_clusters, n_trials=10)
            
            # Store results
            self.results[n_clusters] = {
                'kmeans': kmeans,
                'labels': labels,
                'inertia': inertia
            }
            
            # Create and plot assignment matrix
            assignment_matrix = self.create_assignment_matrix(labels, n_clusters)
            self.plot_assignment_matrix(assignment_matrix, n_clusters, f" ({n_clusters} clusters)")
            
            # Plot centroids
            self.plot_centroids(kmeans, n_clusters, f" ({n_clusters} clusters)")
            
            # Analyze cluster purity
            cluster_analysis = self.analyze_cluster_purity(labels, n_clusters)
            
            # Calculate clustering metrics
            ari = adjusted_rand_score(self.y, labels)
            nmi = normalized_mutual_info_score(self.y, labels)
            
            print(f"\nClustering Metrics:")
            print(f"Inertia: {inertia:.2f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print(f"Normalized Mutual Information: {nmi:.3f}")
            
            # Suggest cluster merging for larger numbers of clusters
            if n_clusters > 10:
                self.suggest_cluster_merging(cluster_analysis)
        
        # Summary comparison
        self.print_summary()
    
    def print_summary(self):
        """Print summary comparison of all clustering results"""
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        
        print(f"{'Clusters':<10} {'Inertia':<12} {'ARI':<8} {'NMI':<8}")
        print(f"{'-'*10} {'-'*12} {'-'*8} {'-'*8}")
        
        for n_clusters in sorted(self.results.keys()):
            inertia = self.results[n_clusters]['inertia']
            ari = adjusted_rand_score(self.y, self.results[n_clusters]['labels'])
            nmi = normalized_mutual_info_score(self.y, self.results[n_clusters]['labels'])
            
            print(f"{n_clusters:<10} {inertia:<12.2f} {ari:<8.3f} {nmi:<8.3f}")
        
        print(f"\nRecommendations:")
        print(f"- 10 clusters: Good baseline, each cluster should ideally represent one digit")
        print(f"- 15-20 clusters: May capture digit variations (e.g., different writing styles)")
        print(f"- 30 clusters: High granularity, multiple clusters per digit - good for sub-digit patterns")

def main():
    """Main function to run the analysis"""
    print("MNIST K-means Clustering Analysis")
    print("=" * 50)
    
    analyzer = EMNISTClusteringAnalysis()
    analyzer.run_complete_analysis()
    
    print(f"\nAnalysis complete! Check the 'img' folder for saved visualizations.")

if __name__ == "__main__":
    main()
