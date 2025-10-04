import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

import scripts.report_portfolio_pnl as rpt


class TestReportPortfolioPnL(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.prev_cwd = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, self.prev_cwd)
        self.addCleanup(self.tmp.cleanup)

    def _write_csv(self, path: Path, rows, header):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def test_build_portfolio_pnl_outputs(self):
        portfolio_rows = [['AAA', 100, 10.0], ['BBB', 50, 20.0]]
        snapshot_rows = [
            ['AAA', 12.0, 'Tech'],
            ['BBB', 18.0, 'Finance'],
            ['VNINDEX', 1200.0, 'Index'],
        ]
        self._write_csv(Path('out/portfolio_clean.csv'), portfolio_rows, ['Ticker','Quantity','AvgCost'])
        self._write_csv(Path('out/snapshot.csv'), snapshot_rows, ['Ticker','Price','Sector'])

        summary_path, sector_path = rpt.build_portfolio_pnl(
            portfolio_csv='out/portfolio_clean.csv',
            snapshot_csv='out/snapshot.csv',
            out_summary_csv='out/portfolio_pnl_summary.csv',
            out_by_sector_csv='out/portfolio_pnl_by_sector.csv',
        )
        summary = list(csv.reader(summary_path.open(encoding='utf-8')))
        self.assertEqual(summary[1], ['2000.0', '2100.0', '100.0', '5.00'])
        by_sector = list(csv.reader(sector_path.open(encoding='utf-8')))
        # header + two sectors
        self.assertEqual(len(by_sector), 3)
        finance_row = next(row for row in by_sector if row[0] == 'Finance')
        self.assertEqual(finance_row[-1], '-10.00')

    def test_fallback_to_industry_map(self):
        portfolio_rows = [['AAA', 100, 10.0]]
        snapshot_rows = [['AAA', 15.0]]  # no sector column
        self._write_csv(Path('out/portfolio_clean.csv'), portfolio_rows, ['Ticker','Quantity','AvgCost'])
        self._write_csv(Path('out/snapshot.csv'), snapshot_rows, ['Ticker','Price'])
        self._write_csv(Path('data/industry_map.csv'), [['AAA','Utilities']], ['Ticker','Sector'])

        _, sector_path = rpt.build_portfolio_pnl(
            portfolio_csv='out/portfolio_clean.csv',
            snapshot_csv='out/snapshot.csv',
            out_summary_csv='out/portfolio_pnl_summary.csv',
            out_by_sector_csv='out/portfolio_pnl_by_sector.csv',
        )
        by_sector = list(csv.reader(sector_path.open(encoding='utf-8')))
        utilities = next(row for row in by_sector if row[0] == 'Utilities')
        self.assertEqual(utilities[0], 'Utilities')

    def test_build_portfolio_pnl_dfs(self):
        portfolio_df = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0},
            {'Ticker': 'BBB', 'Quantity': 50, 'AvgCost': 20.0},
        ])
        snapshot_df = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 6.0, 'Sector': 'Tech'},
            {'Ticker': 'BBB', 'Price': 20.0, 'Sector': 'Finance'},
        ])
        summary_df, by_sector_df = rpt.build_portfolio_pnl_dfs(portfolio_df, snapshot_df)
        self.assertEqual(summary_df.iloc[0]['TotalPnL'], '-400.0')
        self.assertIn('Tech', set(by_sector_df['Sector']))
