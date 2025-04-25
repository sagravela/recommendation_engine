import tensorflow as tf

from src import MODEL_PATH, PROCESSED_PATH


class RecommendationEngine:
    def __init__(self):
        # List of products features used by the model
        self.item_features = [
            "product_id",
            "product_name",
            "category_name",
            "merchant_name",
            "merchant_state",
            "merchant_region",
            "free_shipping",
            "is_sold_out",
            "editor_pick",
            "on_sale",
            "sales_last_week",
            "sales_last_month",
            "sales_last_year",
            "price_in_cents",
            "reviews",
        ]

        index_path = str(MODEL_PATH / "retrieval_index")
        model_path = str(MODEL_PATH / "ranking_model")
        self.index = tf.keras.models.load_model(index_path)
        self.model = tf.keras.models.load_model(model_path)

        products_path = str(PROCESSED_PATH / "products")
        self.products = tf.data.Dataset.load(products_path)

    def get_recommendations(self, raw_query: dict):
        self.query_input = {
            "user_id": tf.convert_to_tensor(raw_query["user_id"], dtype=tf.string),
            "channel": tf.convert_to_tensor(raw_query["channel"], dtype=tf.string),
            "device_type": tf.convert_to_tensor(
                raw_query["device_type"], dtype=tf.string
            ),
            "query_text": tf.convert_to_tensor(
                raw_query["query_text"], dtype=tf.string
            ),
            "time": tf.convert_to_tensor(raw_query["time"].isoformat(), dtype=tf.string),
        }

        # Get recommendations. Note that I am expanding the dimension to match the batch size expected by the model
        _, self.top_rec = self.index({k: [v] for k, v in self.query_input.items()})

        # Filter by product id
        selected_items = self.products.filter(self.filter_by_id)
        # Concat with query input
        recs = selected_items.map(lambda x: {**self.query_input, **x})

        # Get score
        score_added_recs = recs.batch(8).map(self.get_score).unbatch()

        # Order by score
        ordered_recs = self.order_by_score(score_added_recs)

        # Decode values and return
        return list(map(self.parse, ordered_recs))

    def filter_by_id(self, item):
        return tf.reduce_any(tf.equal(item["product_id"], self.top_rec[0]))

    def get_score(self, item):
        # Discard unused features by the model
        input_data = {
            k: v
            for k, v in item.items()
            if k in self.item_features + list(self.query_input.keys())
        }
        _, _, score = self.model(input_data)
        item["score"] = score
        return item

    def order_by_score(self, recs) -> list[dict]:
        rec_list = list(recs.as_numpy_iterator())

        # Descending order by score
        return sorted(rec_list, key=lambda x: x["score"], reverse=True)

    def parse(self, item):
        return {
            "score": float(item["score"][0]),
            "product_name": item["product_name"].decode("utf-8"),
            "category": item["category_name"].decode("utf-8"),
            "price_in_cents":int(item["price_in_cents"]),
            "reviews": int(item["reviews"]),
            "merchant": item["merchant_name"].decode("utf-8"),
            "city": item["merchant_city"].decode("utf-8"),
            "state": item["merchant_state"].decode("utf-8"),
            "region": item["merchant_region"].decode("utf-8"),
            "free_shipping": int(item["free_shipping"]),
            "sold_out": int(item["is_sold_out"]),
            "editors_pick": int(item["editor_pick"]),
            "on_sale": int(item["on_sale"]),
        }
