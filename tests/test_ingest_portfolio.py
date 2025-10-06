from pathlib import Path
import csv
import tempfile
import unittest

from scripts.portfolio.ingest_auto import ingest_portfolio_df


def write_csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Ticker', 'Quantity', 'AvgCost'])
        for r in rows:
            w.writerow(r)


class TestIngestPortfolio(unittest.TestCase):
    def test_ingest_merge_weighted_avg_cost(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            write_csv(dpath / 'pf1.csv', [
                ['AAA', 100, 10.0],
                ['BBB', 200, 20.0],
            ])
            write_csv(dpath / 'pf2.csv', [
                ['AAA', 100, 12.0],
                ['CCC', 50, 30.0],
            ])

            df = ingest_portfolio_df(indir=str(dpath))
            df = df.sort_values('Ticker').reset_index(drop=True)
            a = df[df['Ticker'] == 'AAA'].iloc[0]
            self.assertEqual(int(a['Quantity']), 200)
            self.assertAlmostEqual(float(a['AvgCost']), 11.0)
            self.assertAlmostEqual(float(a['CostValue']), 2200.0)
            b = df[df['Ticker'] == 'BBB'].iloc[0]
            self.assertEqual(int(b['Quantity']), 200)
            self.assertAlmostEqual(float(b['AvgCost']), 20.0)
            c = df[df['Ticker'] == 'CCC'].iloc[0]
            self.assertEqual(int(c['Quantity']), 50)
            self.assertAlmostEqual(float(c['AvgCost']), 30.0)

    def test_ingest_invalid_quantity_raises(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            write_csv(dpath / 'pf1.csv', [
                ['AAA', 'foo', 10.0],
            ])
            with self.assertRaisesRegex(SystemExit, 'invalid Quantity'):
                ingest_portfolio_df(indir=str(dpath))

    def test_ingest_invalid_avgcost_raises(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            write_csv(dpath / 'pf1.csv', [
                ['AAA', 100, ''],
            ])
            with self.assertRaisesRegex(SystemExit, 'invalid AvgCost'):
                ingest_portfolio_df(indir=str(dpath))


if __name__ == '__main__':
    unittest.main()
