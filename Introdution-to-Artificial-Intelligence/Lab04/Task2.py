import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict


class DBSCAN:
    """
    Manual implementation of DBSCAN clustering algorithm
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def _region_query(self, X, point_idx):
        """Find all points within eps distance of point_idx"""
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """Expand cluster from a core point"""
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor is noise, make it part of cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unclassified
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # Find neighbors of this neighbor
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                
                # If neighbor is also a core point, add its neighbors to expansion list
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
                    neighbors = np.unique(neighbors)  # Remove duplicates
            
            i += 1
    
    def fit_predict(self, X):
        """
        Perform DBSCAN clustering
        
        Parameters:
        X: array-like, shape (n_samples, n_features)
        
        Returns:
        labels: array, shape (n_samples,)
            Cluster labels for each point. -1 indicates noise.
        """
        X = np.array(X)
        n_points = len(X)
        
        # Initialize all points as unclassified (0)
        # -1 will be noise, positive integers will be cluster IDs
        labels = np.zeros(n_points, dtype=int)
        cluster_id = 0
        
        for point_idx in range(n_points):
            # Skip if point already processed
            if labels[point_idx] != 0:
                continue
                
            # Find neighbors within eps distance
            neighbors = self._region_query(X, point_idx)
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                # Start new cluster
                cluster_id += 1
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
        
        self.labels_ = labels
        return labels


class DBSCANMNISTAnalyzer:
    def __init__(self, n_samples=12000, pca_components=20):
        self.n_samples = n_samples
        self.pca_components = pca_components
        self.X = None
        self.y = None
        self.X_pca = None
        self.labels = None
        self.best_params = None
        self.best_metrics = None

    def load_data(self):
        """Load and preprocess MNIST data"""
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)

        # Subsample for computational efficiency
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), self.n_samples, replace=False)
        X, y = X[idx], y[idx]

        # Standardize and apply PCA
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=self.pca_components, random_state=42)
        self.X_pca = pca.fit_transform(X_scaled)

        self.X, self.y = X, y
        var = pca.explained_variance_ratio_.sum()
        print(f"Data loaded: {self.n_samples} samples → PCA({self.pca_components}) "
              f"captures {var:.2%} variance.")

    def calculate_silhouette_score(self, X, labels):
        """Calculate silhouette score manually"""
        if len(np.unique(labels[labels != -1])) < 2:
            return 0.0
        
        # Filter out noise points
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        if len(X_filtered) == 0:
            return 0.0
        
        silhouette_scores = []
        
        for i, point in enumerate(X_filtered):
            point_label = labels_filtered[i]
            
            # Calculate intra-cluster distance (a)
            same_cluster_mask = labels_filtered == point_label
            same_cluster_points = X_filtered[same_cluster_mask]
            
            if len(same_cluster_points) > 1:
                a = np.mean(np.sqrt(np.sum((same_cluster_points - point)**2, axis=1)))
            else:
                a = 0
            
            # Calculate inter-cluster distance (b)
            other_clusters = np.unique(labels_filtered[labels_filtered != point_label])
            if len(other_clusters) == 0:
                b = 0
            else:
                min_dist = float('inf')
                for other_label in other_clusters:
                    other_cluster_mask = labels_filtered == other_label
                    other_cluster_points = X_filtered[other_cluster_mask]
                    dist = np.mean(np.sqrt(np.sum((other_cluster_points - point)**2, axis=1)))
                    min_dist = min(min_dist, dist)
                b = min_dist
            
            # Calculate silhouette score for this point
            if max(a, b) > 0:
                s = (b - a) / max(a, b)
            else:
                s = 0
            silhouette_scores.append(s)
        
        return np.mean(silhouette_scores)

    def calculate_metrics(self, labels):
        """Calculate comprehensive clustering metrics"""
        # Basic cluster statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_ratio = n_noise / len(labels)

        # Purity and accuracy calculations
        total_correct = 0
        total_non_noise = 0
        purities = []
        cluster_stats = {}
        
        for cl in set(labels):
            if cl == -1:
                continue
            mask = labels == cl
            counts = Counter(self.y[mask])
            dom, dom_count = counts.most_common(1)[0]
            purity = dom_count / mask.sum()
            purities.append(purity)
            total_correct += dom_count
            total_non_noise += mask.sum()
            
            # Store detailed cluster statistics
            cluster_stats[cl] = {
                'size': mask.sum(),
                'dominant_digit': dom,
                'purity': purity,
                'digit_counts': dict(counts)
            }
            
        avg_purity = np.mean(purities) if purities else 0
        accuracy = total_correct / total_non_noise if total_non_noise else 0
        error_rate = 1 - accuracy

        # Calculate silhouette score
        silhouette = self.calculate_silhouette_score(self.X_pca, labels)

        # Combined score (weighted combination of metrics)
        score = (0.4 * avg_purity +
                 0.3 * accuracy +
                 0.2 * (1 - noise_ratio) +
                 0.1 * silhouette)

        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'purity': avg_purity,
            'accuracy': accuracy,
            'error_rate': error_rate,
            'silhouette': silhouette,
            'combined_score': score,
            'cluster_stats': cluster_stats
        }

    def run(self, eps, min_samples):
        """Run DBSCAN with given parameters"""
        print(f"\nRunning DBSCAN with eps={eps}, min_samples={min_samples}")
        
        # Initialize our manual DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fit and predict
        self.labels = dbscan.fit_predict(self.X_pca)
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.labels)
        
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Noise: {metrics['n_noise']} ({metrics['noise_ratio']:.2%})")
        print(f"Purity: {metrics['purity']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Error rate: {metrics['error_rate']:.4f}")
        print(f"Silhouette: {metrics['silhouette']:.4f}")
        print(f"Combined score: {metrics['combined_score']:.4f}")
        
        return metrics

    def analyze_clusters(self):
        """Provide detailed analysis of clustering results"""
        if self.labels is None:
            print("No clustering has been performed yet.")
            return
        
        digit_totals = Counter(self.y)
        
        print("\nDetailed cluster breakdown:")
        print("-" * 80)
        print(f"{'Cluster':<10}{'Size':<8}{'Dominant':<10}{'Purity':<10}{'Distribution'}")
        print("-" * 80)
        
        # Analyze noise
        noise_mask = self.labels == -1
        if noise_mask.any():
            noise_count = noise_mask.sum()
            noise_dist = Counter(self.y[noise_mask])
            dist_str = ", ".join([f"{d}:{c}" for d, c in noise_dist.most_common(3)])
            print(f"{'Noise':<10}{noise_count:<8}{'N/A':<10}{'N/A':<10}{dist_str}")
        
        # Analyze each cluster
        metrics = self.calculate_metrics(self.labels)
        cluster_stats = metrics['cluster_stats']
        
        for cl in sorted(cluster_stats.keys()):
            stats = cluster_stats[cl]
            dominant = stats['dominant_digit']
            size = stats['size']
            purity = stats['purity']
            
            counts = stats['digit_counts']
            dist_str = ", ".join([f"{d}:{c}" for d, c in 
                                 sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]])
            
            print(f"{cl:<10}{size:<8}{dominant:<10}{purity:.4f}    {dist_str}")
            
        # Digit capture analysis
        print("\nDigit capture analysis:")
        print("-" * 80)
        print(f"{'Digit':<8}{'Total':<8}{'Clustered':<10}{'Noise':<8}{'Capture':<10}{'Accuracy'}")
        print("-" * 80)
        
        digit_clusters = defaultdict(list)
        digit_correct = defaultdict(int)
        digit_in_clusters = defaultdict(int)
        
        # Calculate digit statistics
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
            
            clusters_str = ", ".join([f"{cl}({p:.2f})" for cl, p, _ in digit_clusters[digit]])
            
            print(f"{digit:<8}{total:<8}{clustered:<10}{noise:<8}{capture:.4f}    {accuracy:.4f}")
            if digit_clusters[digit]:
                print(f"         Dominant in clusters: {clusters_str}")
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"Total samples: {len(self.y)}")
        print(f"Clustered samples: {len(self.y) - metrics['n_noise']} ({1-metrics['noise_ratio']:.2%})")
        print(f"Noise samples: {metrics['n_noise']} ({metrics['noise_ratio']:.2%})")
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Average cluster purity: {metrics['purity']:.4f}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"Error rate within clusters: {metrics['error_rate']:.4f}")
        print(f"Silhouette score: {metrics['silhouette']:.4f}")

    def grid_search(self, eps_values, min_samples_values):
        """Perform grid search to find optimal parameters"""
        print(f"Performing grid search over {len(eps_values) * len(min_samples_values)} parameter combinations...")
        
        best_score = -1
        best_params = None
        best_metrics = None
        results = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    metrics = self.run(eps, min_samples)
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        **metrics
                    })
                    
                    if metrics['combined_score'] > best_score:
                        best_score = metrics['combined_score']
                        best_params = (eps, min_samples)
                        best_metrics = metrics
                        
                except Exception as e:
                    print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        if best_params:
            print("\nGrid search complete!")
            print(f"Best parameters: eps={best_params[0]}, min_samples={best_params[1]}")
            print(f"Best score: {best_score:.4f}")
            print(f"Clusters: {best_metrics['n_clusters']}, "
                  f"Noise: {best_metrics['noise_ratio']:.2%}, "
                  f"Purity: {best_metrics['purity']:.4f}, "
                  f"Accuracy: {best_metrics['accuracy']:.4f}")
            
            self.best_params = best_params
            self.best_metrics = best_metrics
            
            # Run with best parameters to set final labels
            self.run(*best_params)
        else:
            print("Grid search failed - no valid results found!")
        
        return results
    


    def generate_report(self):
        """Generate a comprehensive report of the clustering results"""
        if self.labels is None or self.best_metrics is None:
            print("No clustering results available for report generation.")
            return
        
        metrics = self.best_metrics
        
        print("\n" + "="*80)
        print("DBSCAN MNIST CLUSTERING REPORT")
        print("="*80)
        
        print(f"\nBest Parameters:")
        print(f"- eps: {self.best_params[0]}")
        print(f"- min_samples: {self.best_params[1]}")
        
        print(f"\nClustering Results:")
        print(f"- Number of clusters: {metrics['n_clusters']}")
        print(f"- Noise percentage: {metrics['noise_ratio']:.2%} ({metrics['n_noise']} samples)")
        print(f"- Average cluster purity: {metrics['purity']:.4f}")
        print(f"- Classification accuracy: {metrics['accuracy']:.4f}")
        print(f"- Error rate within clusters: {metrics['error_rate']:.4f}")
        print(f"- Silhouette score: {metrics['silhouette']:.4f}")
        
        print(f"\nAnswers to Assignment Questions:")
        print(f"1. Classification accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"2. Noise percentage: {metrics['noise_ratio']:.2%}")
        print(f"3. Error rate in clusters: {metrics['error_rate']:.4f} ({metrics['error_rate']*100:.2f}%)")
        print(f"4. Number of clusters: {metrics['n_clusters']} (target: 10-30)")
        
        if 10 <= metrics['n_clusters'] <= 30:
            print("   ✓ Number of clusters is within acceptable range")
        else:
            print("   ⚠ Number of clusters outside recommended range")
        
        print("="*80)


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = DBSCANMNISTAnalyzer(n_samples=12000, pca_components=20)
    analyzer.load_data()
    
    # Define parameter ranges for grid search
    eps_values = [5.0, 6.0, 15.0]
    min_samples_values = [8, 18, 28]
    
    # Perform grid search
    results = analyzer.grid_search(eps_values, min_samples_values)
    
    # Analyze results
    analyzer.analyze_clusters()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
