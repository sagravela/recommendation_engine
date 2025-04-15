from pathlib import Path

import tensorflow as tf

class RecommendationEngine():
    def __init__(self):
        # List of products features used by the model
        self.model_product_features = [
            'product_id', 'product_name', 'category_name',
            'merchant_name', 'merchant_state', 'merchant_region',
            'free_shipping', 'is_sold_out', 'editor_pick', 'on_sale',
            'sales_last_week', 'sales_last_month', 'sales_last_year',
            'price_in_cents', 'reviews'
        ]

        root_path = Path().resolve().parent
        index_path = str(root_path /  "output" / "model" / "retrieval_index")
        model_path = str(root_path / "output" / "model" / "ranking_model")
        self.index = tf.keras.models.load_model(index_path)
        self.model = tf.keras.models.load_model(model_path)

        products_path = str(root_path / "data" / "processed" / "products")
        self.products = tf.data.Dataset.load(products_path)

    def get_recommendations(self, raw_query: dict):
        self.query_input = {
            'user_id': tf.convert_to_tensor(raw_query['user_id'], dtype=tf.string),
            'channel': tf.convert_to_tensor(raw_query['channel'], dtype=tf.string),
            'device_type': tf.convert_to_tensor(raw_query['device_type'], dtype=tf.string),
            'query_text': tf.convert_to_tensor(raw_query['query_text'], dtype=tf.string),
            'time': tf.convert_to_tensor(raw_query['time'], dtype=tf.string),
        }

        # Get recommendations. Note that I am expanding the dimension to match the batch size expected by the model
        _, self.top_rec = self.index({k: [v] for k, v in self.query_input.items()})

        # Filter by product id
        filtered_recs = self.products.filter(self.filter_by_id)
        # Add query input
        query_added_recs = filtered_recs.map(lambda x: {**self.query_input, **x})

        # Get score
        score_added_recs = query_added_recs.batch(8).map(self.get_score).unbatch()

        # Drop unwanted columns
        recs = score_added_recs.map(self.desired_output)

        # Order by score
        ordered_recs = self.order_by_score(recs)

        # Decode values and return
        return list(map(self.decode_values, ordered_recs))

    def filter_by_id(self, item):
        return tf.reduce_any(tf.equal(item['product_id'], self.top_rec[0]))

    def get_score(self, item):
        # Discard unused features by the model
        input_data = {k: v for k, v in item.items() if k in self.model_product_features + list(self.query_input.keys())}
        _, _, score = self.model(input_data)
        item['score'] = score
        return item

    def desired_output(self, item):
        return {
            'score': item['score'],
            'product_name': item['product_name'],
            'category': item['category_name'],
            'price_in_cents': item['price_in_cents'],
            'reviews': item['reviews'],
            'merchant': item['merchant_name'],
            'city': item['merchant_city'],
            'state': item['merchant_state'],
            'region': item['merchant_region'],
            'free_shipping': item['free_shipping'],
            'sold_out': item['is_sold_out'],
            'editors_pick': item['editor_pick'],
            'on_sale': item['on_sale']
        }

    def order_by_score(self, recs):
        rec_list = list(recs.as_numpy_iterator())

        # Descending order by score
        return sorted(rec_list, key=lambda x: x['score'], reverse=True)

    def decode_values(self, item):
        for key, value in item.items():
            if isinstance(value, bytes):
                item[key] = value.decode('utf-8')
            if key == 'score':
                item[key] = value[0]
        return item