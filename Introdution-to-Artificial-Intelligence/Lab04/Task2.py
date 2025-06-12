import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN  # Używamy sklearn zamiast cl_module
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import os

class DBSCANMNISTAnalyzer:
    def __init__(self, n_samples=12000, pca_components=50):  # Zwiększone PCA
        self.n_samples = n_samples
        self.pca_components = pca_components
        self.X = None
        self.y = None
        self.X_pca = None
        self.labels = None
        self.best_params = None
        self.best_metrics = None

    def load_data(self):
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)

        # subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), self.n_samples, replace=False)
        X, y = X[idx], y[idx]

        # 3) standardize + PCA
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=self.pca_components, random_state=42)
        self.X_pca = pca.fit_transform(X_scaled)

        self.X, self.y = X, y
        var = pca.explained_variance_ratio_.sum()
        print(f"Data loaded: {self.n_samples} samples → PCA({self.pca_components}) "
              f"captures {var:.2%} variance.")

    def calculate_metrics(self, labels):
        # cluster counts
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_ratio = n_noise / len(labels)

        # purity and accuracy
        total_correct = 0
        total_non_noise = 0
        purities = []
        cluster_stats = {}
        
        for cl in set(labels):
            if cl == -1:
                continue
            mask = labels == cl
            counts = Counter(self.y[mask])
            if len(counts) == 0:
                continue
            dom, dom_count = counts.most_common(1)[0]
            purity = dom_count / mask.sum()
            purities.append(purity)
            total_correct += dom_count
            total_non_noise += mask.sum()
            
            # Store cluster statistics
            cluster_stats[cl] = {
                'size': mask.sum(),
                'dominant_digit': dom,
                'purity': purity,
                'digit_counts': dict(counts)
            }
            
        purity = np.mean(purities) if purities else 0
        accuracy = total_correct / total_non_noise if total_non_noise else 0
        error_rate = 1 - accuracy

        # silhouette
        if n_clusters > 1 and n_noise < len(labels):
            try:
                sil = silhouette_score(self.X_pca[labels != -1], labels[labels != -1])
            except:
                sil = 0
        else:
            sil = 0

        # Ulepszona funkcja oceny - kara za bardzo nierównomierne klastry
        cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
        if len(cluster_sizes) > 1:
            size_variance = np.var(cluster_sizes) / np.mean(cluster_sizes)
            balance_penalty = min(0.3, size_variance / 1000)  # Kara za nierównowagę
        else:
            balance_penalty = 0

        # combined score z uwzględnieniem równowagi klastrów
        score = (0.3 * purity +
                 0.3 * accuracy +
                 0.2 * (1 - noise_ratio) +
                 0.1 * sil +
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

    def run(self, eps, min_samples):
        print(f"\nRunning DBSCAN with eps={eps}, min_samples={min_samples}")
        
        # Używamy sklearn DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.labels = db.fit_predict(self.X_pca)
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.labels)
        
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Noise: {metrics['n_noise']} ({metrics['noise_ratio']:.2%})")
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
        if self.labels is None:
            print("No clustering has been performed yet.")
            return
        
        digit_totals = Counter(self.y)
        
        print("\nDetailed cluster breakdown:")
        print("-" * 80)
        print(f"{'Cluster':<8}{'Size':<8}{'%Total':<8}{'Dominant':<10}{'Purity':<10}{'Distribution'}")
        print("-" * 80)
        
        noise_mask = self.labels == -1
        if noise_mask.any():
            noise_count = noise_mask.sum()
            noise_percent = noise_count / len(self.labels) * 100
            noise_dist = Counter(self.y[noise_mask])
            dist_str = ", ".join([f"{d}:{c}" for d, c in noise_dist.most_common(3)])
            print(f"{'Noise':<8}{noise_count:<8}{noise_percent:<7.1f}%{'N/A':<10}{'N/A':<10}{dist_str}")
        
        metrics = self.calculate_metrics(self.labels)
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
                if cl == -1:
                    continue
                mask = (self.labels == cl) & (self.y == digit)
                digit_in_clusters[digit] += mask.sum()
        
        for digit in range(10):
            total = digit_totals[digit]
            clustered = digit_in_clusters[digit]
            noise = total - clustered
            capture = clustered / total if total > 0 else 0
            
            correct = digit_correct[digit]
            accuracy = correct / clustered if clustered > 0 else 0
            
            # Pokaż główne klastry dla tej cyfry
            main_clusters = sorted(digit_clusters[digit], key=lambda x: x[2], reverse=True)[:3]
            clusters_str = ", ".join([f"{cl}({sz})" for cl, p, sz in main_clusters])
            
            print(f"{digit:<8}{total:<8}{clustered:<10}{noise:<8}{capture:<9.3f}{accuracy:<9.3f} {clusters_str}")
        
        print(f"\nSummary:")
        print(f"Total samples: {len(self.y)}")
        print(f"Clustered: {len(self.y) - metrics['n_noise']} ({1-metrics['noise_ratio']:.1%})")
        print(f"Noise: {metrics['n_noise']} ({metrics['noise_ratio']:.1%})")
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Avg purity: {metrics['purity']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Error rate: {metrics['error_rate']:.3f}")

    def grid_search(self, eps_values, min_samples_values):
        print(f"Grid search over {len(eps_values) * len(min_samples_values)} combinations...")
        
        best_score = -1
        best_params = None
        best_metrics = None
        results = []
        
        for i, eps in enumerate(eps_values):
            for j, min_samples in enumerate(min_samples_values):
                print(f"Progress: {i*len(min_samples_values)+j+1}/{len(eps_values)*len(min_samples_values)}")
                
                metrics = self.run(eps, min_samples)
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    **metrics
                })
                
                # Dodatkowe kryteria dla najlepszego wyniku
                valid_result = (
                    metrics['n_clusters'] >= 8 and 
                    metrics['n_clusters'] <= 25 and
                    metrics['noise_ratio'] < 0.4 and
                    metrics['purity'] > 0.6
                )
                
                if valid_result and metrics['combined_score'] > best_score:
                    best_score = metrics['combined_score']
                    best_params = (eps, min_samples)
                    best_metrics = metrics
        
        if best_params is None:
            print("No valid results found, selecting best overall score...")
            best_result = max(results, key=lambda x: x['combined_score'])
            best_params = (best_result['eps'], best_result['min_samples'])
            best_metrics = best_result
        
        print(f"\nBest parameters: eps={best_params[0]}, min_samples={best_params[1]}")
        print(f"Best score: {best_score:.4f}")
        print(f"Clusters: {best_metrics['n_clusters']}, "
              f"Noise: {best_metrics['noise_ratio']:.1%}, "
              f"Purity: {best_metrics['purity']:.3f}, "
              f"Accuracy: {best_metrics['accuracy']:.3f}")
        
        self.best_params = best_params
        self.best_metrics = best_metrics
        
        # Run with best parameters
        self.run(*best_params)
        
        return results

    def plot_results_analysis(self, results):
        """Wizualizacja wyników grid search"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Konwersja wyników do arrays
        eps_vals = [r['eps'] for r in results]
        min_samples_vals = [r['min_samples'] for r in results]
        n_clusters_vals = [r['n_clusters'] for r in results]
        noise_ratios = [r['noise_ratio'] for r in results]
        purities = [r['purity'] for r in results]
        scores = [r['combined_score'] for r in results]
        
        # 1. Heatmapa wyników
        eps_unique = sorted(set(eps_vals))
        min_samples_unique = sorted(set(min_samples_vals))
        
        score_matrix = np.zeros((len(min_samples_unique), len(eps_unique)))
        for r in results:
            i = min_samples_unique.index(r['min_samples'])
            j = eps_unique.index(r['eps'])
            score_matrix[i, j] = r['combined_score']
        
        im = axes[0, 0].imshow(score_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_xticks(range(len(eps_unique)))
        axes[0, 0].set_xticklabels(eps_unique)
        axes[0, 0].set_yticks(range(len(min_samples_unique)))
        axes[0, 0].set_yticklabels(min_samples_unique)
        axes[0, 0].set_xlabel('eps')
        axes[0, 0].set_ylabel('min_samples')
        axes[0, 0].set_title('Combined Score Heatmap')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. Liczba klastrów vs szum (z wyjaśnieniem)
        scatter = axes[0, 1].scatter(n_clusters_vals, noise_ratios, c=purities, 
                                   cmap='viridis', s=60, alpha=0.7)
        axes[0, 1].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, 
                          label='Max acceptable noise (30%)')
        axes[0, 1].axvline(x=10, color='g', linestyle='--', alpha=0.7, 
                          label='Min clusters (10)')
        axes[0, 1].axvline(x=25, color='g', linestyle='--', alpha=0.7, 
                          label='Max clusters (25)')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Noise Ratio')
        axes[0, 1].set_title('Clusters vs Noise (color=purity)')
        axes[0, 1].legend()
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Rozkład wyników
        axes[1, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=max(scores), color='r', linestyle='--', 
                          label=f'Best: {max(scores):.3f}')
        axes[1, 0].set_xlabel('Combined Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Scores')
        axes[1, 0].legend()
        
        # 4. Purity vs Accuracy
        axes[1, 1].scatter(purities, [r['accuracy'] for r in results], 
                          c=noise_ratios, cmap='viridis_r', s=60, alpha=0.7)
        axes[1, 1].set_xlabel('Cluster Purity')
        axes[1, 1].set_ylabel('Classification Accuracy')
        axes[1, 1].set_title('Purity vs Accuracy (color=low noise)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_clusters(self, max_clusters=10, n_examples=8):
        """Ulepszona wizualizacja klastrów"""
        if self.labels is None:
            print("No clustering performed yet.")
            return
        
        # Utwórz folder na obrazy jeśli nie istnieje
        os.makedirs('cluster_images', exist_ok=True)
        
        metrics = self.calculate_metrics(self.labels)
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
            
            title = (f"Cluster {cl}: size={size} ({size/len(self.labels)*100:.1f}%), "
                    f"dominant={dominant}, purity={purity:.2%}")
            plt.suptitle(title, fontsize=12)
            plt.tight_layout()
            
            filename = f"cluster_images/cluster_{cl}_dom{dominant}_size{size}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            
        print(f"Cluster visualizations saved to 'cluster_images/' directory")

def main():
    # Zwiększone parametry dla lepszej analizy
    analyzer = DBSCANMNISTAnalyzer(n_samples=10000, pca_components=50)
    analyzer.load_data()
    
    # Rozszerzone zakresy parametrów
    eps_values = [8.0, 10.0, 12.0, 15.0, 18.0]  # Większe wartości
    min_samples_values = [5, 10, 15, 20, 25]     # Większe wartości
    
    results = analyzer.grid_search(eps_values, min_samples_values)
    
    print("\n" + "="*50)
    print("FINAL ANALYSIS")
    print("="*50)
    
    analyzer.analyze_clusters()
    analyzer.plot_results_analysis(results)
    analyzer.visualize_clusters(max_clusters=8)

if __name__ == "__main__":
    main()
