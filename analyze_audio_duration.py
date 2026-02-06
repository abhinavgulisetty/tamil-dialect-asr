"""Analyze audio duration distribution in training and validation manifests"""

import json

# Simple statistics without matplotlib
def calculate_stats(durations):
    durations = sorted(durations)
    n = len(durations)
    mean = sum(durations) / n
    median = durations[n//2] if n % 2 != 0 else (durations[n//2-1] + durations[n//2]) / 2
    
    percentiles = {}
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(n * p / 100)
        percentiles[p] = durations[min(idx, n-1)]
    
    return {
        'min': min(durations),
        'max': max(durations),
        'mean': mean,
        'median': median,
        'percentiles': percentiles
    }

# Load manifests
with open('train_manifest.json', 'r') as f:
    train_data = [json.loads(line) for line in f]

with open('val_manifest.json', 'r') as f:
    val_data = [json.loads(line) for line in f]

# Extract durations
train_durations = [item['duration'] for item in train_data]
val_durations = [item['duration'] for item in val_data]

# Statistics
def print_stats(durations, name):
    stats = calculate_stats(durations)
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"Total samples: {len(durations)}")
    print(f"Min duration: {stats['min']:.2f}s")
    print(f"Max duration: {stats['max']:.2f}s")
    print(f"Mean duration: {stats['mean']:.2f}s")
    print(f"Median duration: {stats['median']:.2f}s")
    print(f"Total hours: {sum(durations)/3600:.2f}h")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {stats['percentiles'][p]:.2f}s")

print_stats(train_durations, "TRAINING SET")
print_stats(val_durations, "VALIDATION SET")

# Text-based histogram
def text_histogram(durations, name, bins=20):
    min_d, max_d = min(durations), max(durations)
    bin_width = (max_d - min_d) / bins
    
    counts = [0] * bins
    for d in durations:
        bin_idx = min(int((d - min_d) / bin_width), bins - 1)
        counts[bin_idx] += 1
    
    max_count = max(counts)
    print(f"\n{'='*50}")
    print(f"{name} Duration Histogram")
    print(f"{'='*50}")
    
    for i, count in enumerate(counts):
        start = min_d + i * bin_width
        end = start + bin_width
        bar_length = int((count / max_count) * 40)
        bar = '█' * bar_length
        print(f"{start:5.1f}-{end:5.1f}s [{count:4d}] {bar}")

text_histogram(train_durations, "TRAINING SET", bins=15)
text_histogram(val_durations, "VALIDATION SET", bins=15)

# Duration buckets
print(f"\n{'='*50}")
print("Duration Distribution Buckets")
print(f"{'='*50}")

buckets = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, float('inf'))]
for dataset_name, durations in [("Training", train_durations), ("Validation", val_durations)]:
    print(f"\n{dataset_name}:")
    for start, end in buckets:
        count = sum(1 for d in durations if start <= d < end)
        pct = (count / len(durations)) * 100
        end_label = f"{end}s" if end != float('inf') else "∞"
        print(f"  {start}s - {end_label}: {count:4d} samples ({pct:5.1f}%)")
