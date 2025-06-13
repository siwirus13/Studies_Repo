import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import pandas as pd
from collections import Counter, defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('img', exist_ok=True)
os.makedirs('cluster_images', exist_ok=True)

class KMeansEMNISTAnalyzer:
    def __init__(self, n_samples=12000, pca_components=50):
        self.n_samples = n_samples
        self.pca_components = pca_components
        self.X = None
        self.y = None
        self.X_pca = None
        self.X_scaled = None
        self.results = {}
        self.labels = None
        self.best_params = None
        self.best_metrics = None
        
    def load_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)

        # Subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), self.n_samples, replace=False)
        X, y = X[idx], y[idx]

        # Standardize + PCA
        self.X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=self.pca_components, random_state=42)
        self.X_pca = pca.fit_transform(self.X_scaled)

        self.X, self.y = X, y
        var = pca.explained_variance_ratio_.sum()
        print(f"Data loaded: {self.n_samples} samples → PCA({self.pca_components}) "
              f"captures {var:.2%} variance.")
        
    def calculate_metrics(self, labels, n_clusters):
        """Calculate comprehensive clustering metrics"""
        # Basic counts
        n_noise = 0  # K-means doesn't have noise points
        noise_ratio = 0.0
        
        # Purity and accuracy
        total_correct = 0
        total_samples = len(labels)
        purities = []
        cluster_stats = {}
        
        for cl in range(n_clusters):
            mask = labels == cl
            if mask.sum() == 0:
                continue
                
            counts = Counter(self.y[mask])
            if len(counts) == 0:
                continue
                
            dom, dom_count = counts.most_common(1)[0]
            purity = dom_count / mask.sum()
            purities.append(purity)
            total_correct += dom_count
            
            # Store cluster statistics
            cluster_stats[cl] = {
                'size': mask.sum(),
                'dominant_digit': dom,
                'purity': purity,
                'digit_counts': dict(counts)
            }
                
        purity = np.mean(purities) if purities else 0
        accuracy = total_correct / total_samples if total_samples else 0
        error_rate = 1 - accuracy

        # Silhouette score
        try:
            sil = silhouette_score(self.X_pca, labels)
        except:
            sil = 0

        # Balance penalty - kara za bardzo nierównomierne klastry
        cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
        if len(cluster_sizes) > 1:
            size_variance = np.var(cluster_sizes) / np.mean(cluster_sizes)
            balance_penalty = min(0.3, size_variance / 1000)
        else:
            balance_penalty = 0

        # Combined score
        score = (0.4 * purity +
                 0.3 * accuracy +
                 0.2 * sil +
                 0.1 * (1 - balance_penalty))

        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'purity': purity,
            'accuracy': accuracy,
            'error_rate': error_rate,
            'silhouette': sil,
            'combined_score': score,
            'cluster_stats': cluster_stats,
            'balance_penalty': balance_penalty
        }

    def run(self, n_clusters, n_trials=10):
        """Run K-means clustering with multiple trials"""
        print(f"\nRunning K-means with n_clusters={n_clusters}, trials={n_trials}")
        
        best_inertia = float('inf')
        best_kmeans = None
        best_labels = None
        
        for trial in range(n_trials):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                init='k-means++',
                n_init=1,
                max_iter=300,
                random_state=trial,
                algorithm='lloyd'
            )
            
            labels = kmeans.fit_predict(self.X_pca)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
                best_labels = labels
        
        self.labels = best_labels
        
        # Calculate metrics
        metrics = self.calculate_metrics(best_labels, n_clusters)
        metrics['inertia'] = best_inertia
        metrics['kmeans'] = best_kmeans
        
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Inertia: {best_inertia:.2f}")
        print(f"Purity: {metrics['purity']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Error rate: {metrics['error_rate']:.4f}")
        print(f"Silhouette: {metrics['silhouette']:.4f}")
        print(f"Balance penalty: {metrics['balance_penalty']:.4f}")
        print(f"Combined score: {metrics['combined_score']:.4f}")
        
        # Analiza rozkładu wielkości klastrów
        if metrics['cluster_stats']:
            sizes = [stats['size'] for stats in metrics['cluster_stats'].values()]
            print(f"Cluster sizes - min: {min(sizes)}, max: {max(sizes)}, "
                  f"mean: {np.mean(sizes):.1f}, std: {np.std(sizes):.1f}")
        
        return metrics

    def analyze_clusters(self):
        """Detailed cluster analysis"""
        if self.labels is None:
            print("No clustering has been performed yet.")
            return
        
        digit_totals = Counter(self.y)
        n_clusters = len(set(self.labels))
        
        print("\nDetailed cluster breakdown:")
        print("-" * 80)
        print(f"{'Cluster':<8}{'Size':<8}{'%Total':<8}{'Dominant':<10}{'Purity':<10}{'Distribution'}")
        print("-" * 80)
        
        metrics = self.calculate_metrics(self.labels, n_clusters)
        cluster_stats = metrics['cluster_stats']
        
        # Sortuj klastry według rozmiaru (malejąco)
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['size'], reverse=True)
        
        for cl, stats in sorted_clusters:
            dominant = stats['dominant_digit']
            size = stats['size']
            size_percent = size / len(self.labels) * 100
            purity = stats['purity']
            
            counts = stats['digit_counts']
            dist_str = ", ".join([f"{d}:{c}" for d, c in 
                                 sorted(counts.items(), key=lambda x: x[1], reverse=True)[:4]])
            
            print(f"{cl:<8}{size:<8}{size_percent:<7.1f}%{dominant:<10}{purity:<9.3f} {dist_str}")
        
        # Analiza równowagi klastrów
        sizes = [stats['size'] for stats in cluster_stats.values()]
        if len(sizes) > 1:
            print(f"\nCluster balance analysis:")
            print(f"Largest cluster: {max(sizes)} samples ({max(sizes)/len(self.labels)*100:.1f}%)")
            print(f"Smallest cluster: {min(sizes)} samples ({min(sizes)/len(self.labels)*100:.1f}%)")
            print(f"Size ratio (largest/smallest): {max(sizes)/min(sizes):.1f}")
            print(f"Coefficient of variation: {np.std(sizes)/np.mean(sizes):.3f}")
        
        self._print_digit_analysis(cluster_stats, digit_totals, metrics)

    def _print_digit_analysis(self, cluster_stats, digit_totals, metrics):
        """Print detailed digit analysis"""
        print("\nDigit capture analysis:")
        print("-" * 90)
        print(f"{'Digit':<8}{'Total':<8}{'Clustered':<10}{'Noise':<8}{'Capture':<10}{'Accuracy':<10}{'Main Clusters'}")
        print("-" * 90)
        
        digit_clusters = defaultdict(list)
        digit_correct = defaultdict(int)
        digit_in_clusters = defaultdict(int)
        
        for cl, stats in cluster_stats.items():
            dom_digit = stats['dominant_digit']
            digit_clusters[dom_digit].append((cl, stats['purity'], stats['size']))
            digit_correct[dom_digit] += int(stats['size'] * stats['purity'])
            
        for digit in range(10):
            for cl in set(self.labels):
                mask = (self.labels == cl) & (self.y == digit)
                digit_in_clusters[digit] += mask.sum()
        
        for digit in range(10):
            total = digit_totals[digit]
            clustered = digit_in_clusters[digit]
            noise = 0  # K-means doesn't have noise
            capture = clustered / total if total > 0 else 0
            
            correct = digit_correct[digit]
            accuracy = correct / clustered if clustered > 0 else 0
            
            # Pokaż główne klastry dla tej cyfry
            main_clusters = sorted(digit_clusters[digit], key=lambda x: x[2], reverse=True)[:3]
            clusters_str = ", ".join([f"{cl}({sz})" for cl, p, sz in main_clusters])
            
            print(f"{digit:<8}{total:<8}{clustered:<10}{noise:<8}{capture:<9.3f}{accuracy:<9.3f} {clusters_str}")
        
        print(f"\nSummary:")
        print(f"Total samples: {len(self.y)}")
        print(f"Clustered: {len(self.y)} (100%)")  # K-means clusters all points
        print(f"Noise: 0 (0%)")  # K-means doesn't have noise
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Avg purity: {metrics['purity']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Error rate: {metrics['error_rate']:.3f}")

    def grid_search(self, cluster_values, n_trials_per_config=5):
        """Grid search over different numbers of clusters"""
        print(f"Grid search over {len(cluster_values)} cluster configurations...")
        
        best_score = -1
        best_params = None
        best_metrics = None
        results = []
        
        for i, n_clusters in enumerate(cluster_values):
            print(f"Progress: {i+1}/{len(cluster_values)}")
            
            metrics = self.run(n_clusters, n_trials_per_config)
            
            results.append({
                'n_clusters': n_clusters,
                **metrics
            })
            
            # Kryteria dla najlepszego wyniku
            valid_result = (
                metrics['n_clusters'] >= 8 and 
                metrics['n_clusters'] <= 35 and
                metrics['purity'] > 0.5
            )
            
            if valid_result and metrics['combined_score'] > best_score:
                best_score = metrics['combined_score']
                best_params = n_clusters
                best_metrics = metrics
        
        if best_params is None:
            print("No valid results found, selecting best overall score...")
            best_result = max(results, key=lambda x: x['combined_score'])
            best_params = best_result['n_clusters']
            best_metrics = best_result
        
        print(f"\nBest parameters: n_clusters={best_params}")
        print(f"Best score: {best_score:.4f}")
        print(f"Clusters: {best_metrics['n_clusters']}, "
              f"Purity: {best_metrics['purity']:.3f}, "
              f"Accuracy: {best_metrics['accuracy']:.3f}")
        
        self.best_params = best_params
        self.best_metrics = best_metrics
        
        # Run with best parameters
        self.run(best_params)
        
        return results

    def plot_results_analysis(self, results):
        """Wizualizacja wyników grid search - podobna do DBSCAN"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Konwersja wyników do arrays
        n_clusters_vals = [r['n_clusters'] for r in results]
        purities = [r['purity'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        scores = [r['combined_score'] for r in results]
        silhouettes = [r['silhouette'] for r in results]
        inertias = [r['inertia'] for r in results]
        
        # 1. Wyniki vs liczba klastrów
        ax1 = axes[0, 0]
        ax1.plot(n_clusters_vals, scores, 'bo-', label='Combined Score', alpha=0.7)
        ax1.plot(n_clusters_vals, purities, 'ro-', label='Purity', alpha=0.7)
        ax1.plot(n_clusters_vals, accuracies, 'go-', label='Accuracy', alpha=0.7)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Score')
        ax1.set_title('K-means: Performance vs Number of Clusters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Purity vs Accuracy (podobnie jak w DBSCAN)
        scatter = axes[0, 1].scatter(purities, accuracies, c=n_clusters_vals, 
                                   cmap='viridis', s=80, alpha=0.7)
        axes[0, 1].set_xlabel('Cluster Purity')
        axes[0, 1].set_ylabel('Classification Accuracy')
        axes[0, 1].set_title('K-means: Purity vs Accuracy (color=clusters)')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Rozkład wyników (podobnie jak w DBSCAN)
        axes[1, 0].hist(scores, bins=min(15, len(scores)), alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=max(scores), color='r', linestyle='--', 
                          label=f'Best: {max(scores):.3f}')
        axes[1, 0].set_xlabel('Combined Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('K-means: Distribution of Scores')
        axes[1, 0].legend()
        
        # 4. Inertia vs Silhouette
        axes[1, 1].scatter(inertias, silhouettes, c=n_clusters_vals, 
                          cmap='viridis', s=80, alpha=0.7)
        axes[1, 1].set_xlabel('Inertia (lower is better)')
        axes[1, 1].set_ylabel('Silhouette Score (higher is better)')
        axes[1, 1].set_title('K-means: Inertia vs Silhouette (color=clusters)')
        
        plt.tight_layout()
        
        # Zapisz wykres
        plt.savefig('img/kmeans_grid_search_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_clusters(self, max_clusters=10, n_examples=8):
        """Wizualizacja klastrów - identyczna jak w DBSCAN"""
        if self.labels is None:
            print("No clustering performed yet.")
            return
        
        n_clusters = len(set(self.labels))
        metrics = self.calculate_metrics(self.labels, n_clusters)
        cluster_stats = metrics['cluster_stats']
        
        # Sortuj klastry według rozmiaru
        sorted_clusters = sorted(cluster_stats.items(), 
                               key=lambda x: x[1]['size'], reverse=True)
        
        clusters_to_show = sorted_clusters[:max_clusters]
        
        for cl, stats in clusters_to_show:
            dominant = stats['dominant_digit']
            purity = stats['purity']
            size = stats['size']
            
            indices = np.where(self.labels == cl)[0]
            sample_indices = np.random.choice(indices, 
                                            size=min(n_examples, len(indices)), 
                                            replace=False)
            
            # Grupuj przykłady według prawdziwej cyfry
            examples_by_digit = defaultdict(list)
            for idx in sample_indices:
                examples_by_digit[self.y[idx]].append(idx)
            
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            
            plot_idx = 0
            for digit in sorted(examples_by_digit.keys()):
                if plot_idx >= 8:
                    break
                    
                examples = examples_by_digit[digit]
                for idx in examples[:1]:  # Jeden przykład na cyfrę
                    if plot_idx < 8:
                        axes[plot_idx].imshow(self.X[idx].reshape(28, 28), cmap='gray')
                        axes[plot_idx].set_title(f"True: {self.y[idx]}")
                        axes[plot_idx].axis('off')
                        plot_idx += 1
            
            # Ukryj nieużywane subploty
            for i in range(plot_idx, 8):
                axes[i].axis('off')
            
            title = (f"K-means Cluster {cl}: size={size} ({size/len(self.labels)*100:.1f}%), "
                    f"dominant={dominant}, purity={purity:.2%}")
            plt.suptitle(title, fontsize=12)
            plt.tight_layout()
            
            filename = f"cluster_images/kmeans_cluster_{cl}_dom{dominant}_size{size}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            
        print(f"K-means cluster visualizations saved to 'cluster_images/' directory")

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

    def run_complete_analysis(self):
        """Run complete analysis similar to original task requirements"""
        self.load_data()
        
        cluster_numbers = [10, 15, 20, 30]
        
        print(f"\n{'='*80}")
        print("K-MEANS CLUSTERING ANALYSIS ON MNIST DATASET")
        print(f"{'='*80}")
        
        all_results = []
        
        for n_clusters in cluster_numbers:
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR {n_clusters} CLUSTERS")
            print(f"{'='*60}")
            
            # Perform clustering
            metrics = self.run(n_clusters, n_trials=10)
            all_results.append(metrics)
            
            # Store results
            self.results[n_clusters] = metrics
            
            # Create and plot assignment matrix
            assignment_matrix = self.create_assignment_matrix(self.labels, n_clusters)
            self.plot_assignment_matrix(assignment_matrix, n_clusters, f" ({n_clusters} clusters)")
            
            # Plot centroids
            self.plot_centroids(metrics['kmeans'], n_clusters, f" ({n_clusters} clusters)")
            
            # Analyze clusters
            self.analyze_clusters()
            
            print(f"\nClustering Metrics:")
            print(f"Inertia: {metrics['inertia']:.2f}")
            print(f"Adjusted Rand Index: {adjusted_rand_score(self.y, self.labels):.3f}")
            print(f"Normalized Mutual Information: {normalized_mutual_info_score(self.y, self.labels):.3f}")
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
        
        # Summary comparison
        self.print_summary()
        
        return all_results
    
    def print_summary(self):
        """Print summary comparison of all clustering results"""
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        
        print(f"{'Clusters':<10} {'Inertia':<12} {'Purity':<8} {'Accuracy':<10} {'Silhouette':<12} {'Score':<8}")
        print(f"{'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")
        
        for n_clusters in sorted(self.results.keys()):
            metrics = self.results[n_clusters]
            
            print(f"{n_clusters:<10} {metrics['inertia']:<12.2f} {metrics['purity']:<8.3f} "
                  f"{metrics['accuracy']:<10.3f} {metrics['silhouette']:<12.3f} {metrics['combined_score']:<8.3f}")
        
        print(f"\nRecommendations:")
        print(f"- 10 clusters: Baseline - each cluster represents one digit")
        print(f"- 15-20 clusters: Captures digit variations and writing styles")
        print(f"- 30 clusters: High granularity - multiple sub-patterns per digit")

def main():
    """Main function following DBSCAN style analysis"""
    # Similar parameters to DBSCAN for comparison
    analyzer = KMeansEMNISTAnalyzer(n_samples=12000, pca_components=50)
    analyzer.load_data()
    
    # Test different numbers of clusters
    cluster_values = [8, 10, 12, 15, 18, 20, 25, 30]
    
    results = analyzer.grid_search(cluster_values, n_trials_per_config=5)
    
    print("\n" + "="*50)
    print("FINAL ANALYSIS")
    print("="*50)
    
    analyzer.analyze_clusters()
    analyzer.plot_results_analysis(results)
    analyzer.visualize_clusters(max_clusters=8)
    
    # Run the complete original analysis
    print("\n" + "="*50)
    print("ORIGINAL TASK ANALYSIS (10, 15, 20, 30 clusters)")
    print("="*50)
    
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
