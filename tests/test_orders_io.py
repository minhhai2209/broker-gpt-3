import unittest
import pandas as pd

from scripts.order_engine import Order
from scripts.orders_io import print_orders


class TestOrdersIO(unittest.TestCase):
    def test_print_orders_contains_core_fields(self):
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0}
        ])
        orders = [Order(ticker='AAA', side='BUY', quantity=200, limit_price=19.95, note='Mua mới')]
        out = print_orders(orders, snapshot)
        self.assertIn('AAA', out)
        self.assertIn('19.95', out)
        self.assertIn('Giá thị trường: 20.00', out)
        self.assertIn('Tổng mua', out)


if __name__ == '__main__':
    unittest.main()
