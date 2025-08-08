"""
Lightweight clustering and math utilities - Pure Python implementation
Replaces numpy and scikit-learn dependencies
"""
import math
from datetime import datetime
import logging
from django.utils import timezone
from zoneinfo import ZoneInfo

# Set up logging
logger = logging.getLogger(__name__)


class SimpleMath:
    """Simple math operations to replace numpy"""

    @staticmethod
    def mean(values):
        """Calculate mean of a list of values"""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def average(values, weights=None):
        """Calculate weighted average"""
        if not values:
            return 0.0
        if weights is None:
            return SimpleMath.mean(values)

        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum != 0 else 0.0

    @staticmethod
    def distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


class SimpleDBSCAN:
    """Lightweight DBSCAN clustering implementation"""

    def __init__(self, eps=0.01, min_samples=2):
        # min_samples is interpreted in the conventional way:
        # minimum number of points in the neighborhood including the point itself.
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, points):
        """Perform DBSCAN clustering"""
        n_points = len(points)
        labels = [-1] * n_points  # -1 means noise
        cluster_id = 0
        visited = [False] * n_points

        for i in range(n_points):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(points, i)

            # include the point itself when checking min_samples
            if (len(neighbors) + 1) < self.min_samples:
                # Point is noise (not enough neighbors including itself)
                continue

            # Start new cluster
            labels[i] = cluster_id

            # Use a queue/set for seed expansion (avoid duplicating neighbor indices)
            seed_set = list(neighbors)
            seed_index = 0
            seen = set(seed_set)
            while seed_index < len(seed_set):
                neighbor_idx = seed_set[seed_index]

                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    new_neighbors = self._get_neighbors(points, neighbor_idx)

                    # check new_neighbors including the neighbor itself
                    if (len(new_neighbors) + 1) >= self.min_samples:
                        # add any new neighbor indices not already seen
                        for nn in new_neighbors:
                            if nn not in seen:
                                seen.add(nn)
                                seed_set.append(nn)

                # If neighbor was noise (label -1) or unassigned, assign to current cluster
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

                seed_index += 1

            cluster_id += 1

        return labels

    def _get_neighbors(self, points, point_idx):
        """Get neighbors within eps distance (excluding the point itself)"""
        neighbors = []
        current_point = points[point_idx]

        for i, point in enumerate(points):
            if i == point_idx:
                continue
            if SimpleMath.distance(current_point, point) <= self.eps:
                neighbors.append(i)

        return neighbors


def convert_types(obj):
    """Convert any special types to basic Python types"""
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_types(item) for item in obj)
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        return str(obj)


def format_timestamp_ist(dt):
    """Convert datetime to IST and return ISO format string"""
    if not dt:
        return None

    # Convert to IST timezone using zoneinfo (built into Python 3.9+)
    ist_tz = ZoneInfo('Asia/Kolkata')
    if timezone.is_aware(dt):
        ist_time = dt.astimezone(ist_tz)
    else:
        # If naive, assume it's UTC and make it aware
        utc_time = dt.replace(tzinfo=ZoneInfo('UTC'))
        ist_time = utc_time.astimezone(ist_tz)

    return ist_time.isoformat()


def parse_datetime(dt_string):
    """Parse datetime string to datetime object"""
    if isinstance(dt_string, str):
        try:
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except ValueError:
            # fallback format
            return timezone.make_aware(datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S'))
    return dt_string


def process_kalman_cluster_fusion(event_data):
    """
    Process event data using simplified clustering algorithm
    Each user is processed independently with their own clustering
    Returns analysis results
    """
    try:
        if not event_data:
            return {'error': 'No data to process'}

        logger.info(f"Processing {len(event_data)} events")

        # Convert timestamp strings to datetime objects and ensure awareness
        for event in event_data:
            event['timestamp'] = parse_datetime(event['timestamp'])
            if event['timestamp'].tzinfo is None:
                event['timestamp'] = timezone.make_aware(event['timestamp'])

        # Event weights (updated to equal weights for all events)
        event_weights = {
            'upi': 1.0,
            'app_open': 1.0,
            'login': 1.0
        }

        # Step 1: Group events by user (each user processed independently)
        user_events = {}
        for event in event_data:
            user_id = event['user_id']
            if user_id not in user_events:
                user_events[user_id] = []
            user_events[user_id].append(event)

        logger.info(f"Processing {len(user_events)} users independently")

        # Step 2: Process each user independently with their own clustering
        user_results = []
        all_user_clusters = {}  # Store cluster info for each user
        total_clusters_count = 0
        total_noise_points = 0

        for user_id, events in user_events.items():
            logger.info(f"Processing user {user_id} with {len(events)} events")

            # Perform clustering for this user's events only
            user_coordinates = [[event['lat'], event['lon']] for event in events]

            # Use eps=0.01 degrees (roughly 1.1 km) for clustering
            dbscan = SimpleDBSCAN(eps=0.01, min_samples=2)
            user_clusters = dbscan.fit_predict(user_coordinates)

            # Add cluster info to this user's events (with user-specific cluster IDs)
            for i, event in enumerate(events):
                if user_clusters[i] == -1:
                    event['cluster'] = -1  # Noise
                    total_noise_points += 1
                else:
                    # Create unique cluster ID: user_id_clusternum
                    event['cluster'] = f"{user_id}_{user_clusters[i]}"

            # Count clusters for this user
            user_unique_clusters = set(user_clusters)
            user_cluster_count = len(user_unique_clusters) - (1 if -1 in user_unique_clusters else 0)
            total_clusters_count += user_cluster_count

            logger.info(f"User {user_id}: Found {user_cluster_count} clusters")

            # Process this user's data (this function will compute weights & cluster predictions using those weights)
            user_result = process_user_data_independent(events, event_weights, user_id)
            user_results.append(user_result)

            # Store cluster info for this user
            all_user_clusters[user_id] = get_user_cluster_info(events, user_id)

        # Step 3: Generate summary statistics across all users
        # Event type distribution
        event_types = {}
        for event in event_data:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Global cluster distribution (per user)
        user_cluster_dist = {}
        for user_id, user_cluster_info in all_user_clusters.items():
            user_cluster_count = len(user_cluster_info)
            user_cluster_dist[user_id] = user_cluster_count

        summary = {
            'total_users': len(user_events),
            'total_events': len(event_data),
            'total_clusters': total_clusters_count,
            'noise_points': total_noise_points,
            'event_distribution': event_types,
            'user_cluster_distribution': user_cluster_dist,
            'processing_timestamp': format_timestamp_ist(timezone.now()),
            'anomalies': detect_user_anomalies(user_results),
            'confidence': calculate_overall_confidence(user_results)
        }

        result = {
            'summary': summary,
            'user_results': user_results,
            'all_user_clusters': all_user_clusters,
            'location_predictions': generate_user_location_predictions(user_results),
            'algorithm_parameters': {
                'dbscan_eps': 0.01,
                'dbscan_min_samples': 2,
                'event_weights': event_weights,
                'time_decay_hours_primary': 72,
                'time_decay_hours_fallback': 336,
                'night_boost_factor': 1.2,
                'independent_user_processing': True
            }
        }

        logger.info("Processing completed successfully")
        # Convert any remaining special types
        return convert_types(result)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {'error': f'Processing failed: {str(e)}'}


def process_user_data_independent(user_events, event_weights, user_id):
    """Process individual user data independently with their own clustering"""
    if not user_events:
        return None

    # Calculate event weights with smart time decay
    current_time = timezone.now()
    weighted_events = []
    
    # Determine appropriate time decay window
    # Check if there are any events within the last 72 hours
    recent_events = []
    for event in user_events:
        event_time = event['timestamp']
        if event_time.tzinfo is None:
            event_time = timezone.make_aware(event_time)
        
        time_diff_hours = (current_time - event_time).total_seconds() / 3600
        if time_diff_hours <= 72:
            recent_events.append({
                'event_type': event['event_type'],
                'hours_ago': time_diff_hours
            })
    
    # Use 72 hours if we have ANY events within 72h, otherwise use 336 hours (2 weeks)
    # Rule: If NO events in 72hrs, then shift to 336hrs
    time_decay_window = 72 if len(recent_events) > 0 else 336
    logger.info(f"User {user_id}: Using {time_decay_window}h time decay window (recent events: {len(recent_events)})")
    
    # Debug: Log recent events details
    if recent_events:
        recent_details = [f"{e['event_type']}({e['hours_ago']:.1f}h)" for e in recent_events]
        logger.info(f"User {user_id}: Recent events within 72h: {', '.join(recent_details)}")
    else:
        logger.info(f"User {user_id}: No events within 72h, using extended window")

    for event in user_events:
        base_weight = event_weights.get(event['event_type'], 1.0)

        # Time decay with smart window - handle timezone properly
        event_time = event['timestamp']
        if event_time.tzinfo is None:
            event_time = timezone.make_aware(event_time)

        time_diff = (current_time - event_time).total_seconds() / 3600  # hours
        time_decay = max(0.0, 1 - (time_diff / time_decay_window))

        # Night boost (assume 22:00 - 06:00 is night)
        event_hour = event_time.hour
        night_boost = 1.2 if (event_hour >= 22 or event_hour <= 6) else 1.0

        final_weight = base_weight * time_decay * night_boost

        # Annotate weighted events (keep cluster id as set earlier)
        weighted_events.append({
            'event_type': event['event_type'],
            'timestamp': event_time.isoformat(),
            'lat': event['lat'],
            'lon': event['lon'],
            'cluster': event.get('cluster', -1),
            'base_weight': base_weight,
            'time_decay': time_decay,
            'time_decay_window': time_decay_window,
            'night_boost': night_boost,
            'final_weight': final_weight
        })

    # Use weighted_events for cluster-based predictions (so we use final_weight consistently)
    clustered_events = [e for e in weighted_events if e['cluster'] != -1]

    if clustered_events:
        # Group by cluster and calculate weighted centroids using final_weight
        cluster_groups = {}
        for event in clustered_events:
            cluster_id = event['cluster']
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(event)

        cluster_predictions = []
        for cluster_id, cluster_events in cluster_groups.items():
            weights = [max(0.0, e.get('final_weight', 0.0)) for e in cluster_events]
            lats = [e['lat'] for e in cluster_events]
            lons = [e['lon'] for e in cluster_events]

            # If all weights are zero (older events), fall back to equal weighting
            if sum(weights) == 0:
                weights = [1.0] * len(cluster_events)

            weighted_lat = SimpleMath.average(lats, weights)
            weighted_lon = SimpleMath.average(lons, weights)

            # Improved confidence calculation
            # Base confidence on time decay and event count
            avg_time_decay = SimpleMath.mean([e.get('time_decay', 0.0) for e in cluster_events])
            event_count_factor = min(1.0, len(cluster_events) / 5.0)  # Normalize to max 5 events
            
            # Calculate confidence as combination of:
            # 1. Average time decay (how recent the events are)
            # 2. Event count factor (more events = higher confidence)
            # 3. Weight consistency (how similar the weights are)
            weight_consistency = 1.0 - (max(weights) - min(weights)) / max(weights) if max(weights) > 0 else 1.0
            
            confidence = (avg_time_decay * 0.5 + event_count_factor * 0.3 + weight_consistency * 0.2)
            confidence = float(max(0.1, min(1.0, confidence)))  # Ensure confidence is between 0.1 and 1.0

            cluster_predictions.append({
                'cluster_id': str(cluster_id),
                'predicted_lat': float(weighted_lat),
                'predicted_lon': float(weighted_lon),
                'event_count': len(cluster_events),
                'confidence': confidence,
                'avg_time_decay': float(avg_time_decay),
                'weight_consistency': float(weight_consistency)
            })

        # Primary prediction (highest confidence cluster)
        if cluster_predictions:
            primary_prediction = max(cluster_predictions, key=lambda x: x['confidence'])
        else:
            primary_prediction = None
    else:
        cluster_predictions = []
        primary_prediction = None

    # Event type counts (from raw user_events)
    event_types = {}
    for event in user_events:
        event_type = event['event_type']
        event_types[event_type] = event_types.get(event_type, 0) + 1

    # Cluster involvement (use cluster values present in weighted_events to be consistent)
    unique_clusters = set(e['cluster'] for e in weighted_events)
    clusters_involved = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    # Time range (from original user_events datetimes)
    timestamps = [e['timestamp'] for e in user_events if e.get('timestamp') is not None]
    first_event = min(timestamps).isoformat() if timestamps else None
    last_event = max(timestamps).isoformat() if timestamps else None

    result = {
        'user_id': user_id,
        'total_events': len(user_events),
        'event_types': event_types,
        'clusters_involved': clusters_involved,
        'weighted_events': weighted_events,
        'cluster_predictions': cluster_predictions,
        'primary_prediction': primary_prediction,
        'time_range': {
            'first_event': first_event,
            'last_event': last_event
        }
    }

    return convert_types(result)


def get_user_cluster_info(user_events, user_id):
    """Get detailed cluster information for a specific user"""
    cluster_info = {}

    # Group events by cluster for this user
    cluster_groups = {}
    for event in user_events:
        cluster_id = event['cluster']
        if cluster_id == -1:
            continue
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(event)

    for cluster_id, cluster_events in cluster_groups.items():
        # Event types in this cluster
        event_types = {}
        for event in cluster_events:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Calculate centroid (simple mean)
        lats = [event['lat'] for event in cluster_events]
        lons = [event['lon'] for event in cluster_events]

        cluster_info[str(cluster_id)] = {
            'cluster_id': str(cluster_id),
            'user_id': user_id,
            'event_count': len(cluster_events),
            'event_types': event_types,
            'centroid': {
                'lat': float(SimpleMath.mean(lats)) if lats else 0.0,
                'lon': float(SimpleMath.mean(lons)) if lons else 0.0
            },
            'bounding_box': {
                'min_lat': float(min(lats)) if lats else 0.0,
                'max_lat': float(max(lats)) if lats else 0.0,
                'min_lon': float(min(lons)) if lons else 0.0,
                'max_lon': float(max(lons)) if lons else 0.0
            }
        }

    return convert_types(cluster_info)


def detect_user_anomalies(user_results):
    """Detect anomalous patterns in user data"""
    anomalies = []

    for result in user_results:
        if not result:
            continue

        user_id = result['user_id']

        # Check for users with many events but no clusters
        if result['total_events'] >= 5 and result['clusters_involved'] == 0:
            anomalies.append({
                'type': 'no_clusters',
                'user_id': user_id,
                'event_count': result['total_events'],
                'description': f'User {user_id} has {result["total_events"]} events but no clusters'
            })

        # Check for users with very high activity
        if result['total_events'] > 50:
            anomalies.append({
                'type': 'high_activity',
                'user_id': user_id,
                'event_count': result['total_events'],
                'description': f'User {user_id} has unusually high activity: {result["total_events"]} events'
            })

        # Check for users with only UPI events but multiple clusters
        if (result['clusters_involved'] > 2 and
                len(result['event_types']) == 1 and
                'upi' in result['event_types']):
            anomalies.append({
                'type': 'scattered_upi',
                'user_id': user_id,
                'clusters': result['clusters_involved'],
                'description': f'User {user_id} has UPI events scattered across {result["clusters_involved"]} clusters'
            })

    return anomalies


def generate_user_location_predictions(user_results):
    """Generate location predictions for all users based on their independent analysis"""
    predictions = {}

    for result in user_results:
        if not result:
            continue

        user_id = result['user_id']
        if result.get('primary_prediction'):
            pred = result['primary_prediction']
            predictions[user_id] = {
                'predicted_lat': pred['predicted_lat'],
                'predicted_lon': pred['predicted_lon'],
                'confidence': pred['confidence'],
                'cluster_id': pred['cluster_id'],
                'event_count': pred['event_count'],
                'prediction_type': 'user_cluster_based',
                'timestamp': format_timestamp_ist(timezone.now())
            }
        else:
            # Fallback to simple average if no clusters for this user
            if result.get('weighted_events'):
                events = result['weighted_events']
                lats = [event['lat'] for event in events]
                lons = [event['lon'] for event in events]
                predictions[user_id] = {
                    'predicted_lat': float(SimpleMath.mean(lats)) if lats else 0.0,
                    'predicted_lon': float(SimpleMath.mean(lons)) if lons else 0.0,
                    'confidence': 0.3,  # Low confidence for non-clustered data
                    'cluster_id': None,
                    'event_count': len(events),
                    'prediction_type': 'user_simple_average',
                    'timestamp': format_timestamp_ist(timezone.now())
                }

    return convert_types(predictions)


def calculate_overall_confidence(user_results):
    """Calculate overall confidence score for the analysis"""
    if not user_results:
        return 0.0

    confidences = []
    for result in user_results:
        if result and result.get('primary_prediction'):
            confidences.append(result['primary_prediction']['confidence'])
        else:
            confidences.append(0.0)

    return float(SimpleMath.mean(confidences)) if confidences else 0.0


def get_user_prediction(user_id, processed_results):
    """Get processed location prediction for a specific user"""
    if not processed_results or 'location_predictions' not in processed_results:
        return None

    predictions = processed_results['location_predictions']
    return predictions.get(user_id, None)
