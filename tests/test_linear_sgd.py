"""Behavioral checks for leakage, checkpoints, divergence and legacy routing."""

import tempfile
import csv
import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.approach_1_linear_sgd.wrapper import LinearSGDWrapper
from src.approach_1_linear_sgd.trainer import score
from src.approach_1_linear_sgd.model import LinearRegressionModel
from src.runner import _fit_model
from src.models import get_model_spec
from src.config import EXPERIMENTS_SPLITS_PATH
from src.approach_1_linear_sgd import runner as approach_runner
from src.approach_1_linear_sgd.pipeline import LinearSGDPipeline
from src.features import build_feature_recipe
from src.config import PROJECT_ROOT, RAW_NUM_COLS
from src.experiment_io import file_lock
from src.approach_1_linear_sgd.logging import run_lock_path, sync_existing_run
from src.leaderboard import build_leaderboard


class CountingRecipe:
    """Picklable stand-in for orchestration tests; validates label isolation on reload."""
    def __init__(self):
        self.transform_calls = 0

    def fit_transform(self, rows, y):
        return np.array([row['features'] for row in rows], dtype=np.float32)

    def transform(self, rows):
        self.transform_calls += 1
        return np.array([row['features'] for row in rows], dtype=np.float32)


class LinearSGDTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.xt = rng.normal(size=(100, 4)).astype(np.float32)
        self.xv = rng.normal(loc=3, size=(20, 4)).astype(np.float32)
        self.yt = 5 + self.xt[:, 0] * 0.3
        self.yv = 5 + self.xv[:, 0] * 0.3

    def test_sparse_train_only_scaling_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = LinearSGDWrapper(max_epochs=15, num_threads=2)
            summary = reg.fit(csr_matrix(self.xt), self.yt, csr_matrix(self.xv), self.yv, tmp, "check")
            self.assertEqual(reg.scaler.n_samples_seen_, len(self.xt))
            np.testing.assert_allclose(reg.scaler.mean_, self.xt.mean(axis=0), atol=1e-7)
            actual = score(self.yv, reg.predict(csr_matrix(self.xv)))
            self.assertLess(abs(actual['rmse'] - summary['best_val_rmse']), 1e-5)
            saved = torch.load(summary['checkpoint_path'], weights_only=True)
            model = LinearRegressionModel(4)
            model.load_state_dict(saved['state_dict'])
            with torch.no_grad():
                pred = model(torch.from_numpy(reg.scaler.transform(self.xv).astype(np.float32))).numpy().ravel()
            np.testing.assert_array_equal(pred, reg.predict(self.xv))
            again = LinearSGDWrapper(max_epochs=15, num_threads=2)
            again.fit(self.xt, self.yt, self.xv, self.yv, tmp, "repeat")
            np.testing.assert_array_equal(reg.predict(self.xv), again.predict(self.xv))

    def test_divergence_retains_best_checkpoint(self):
        original = torch.optim.SGD.step
        calls = 0

        def explode(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(optimizer, *args, **kwargs)
            if calls == 2:
                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.fill_(float('inf'))
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.optim.SGD, 'step', explode):
            reg = LinearSGDWrapper(max_epochs=5)
            summary = reg.fit(self.xt, self.yt, self.xv, self.yv, tmp, "diverge")
            self.assertEqual(summary['status'], 'diverged')
            self.assertEqual(summary['best_epoch'], 1)
            self.assertTrue(np.isfinite(reg.predict(self.xv)).all())
            self.assertTrue(Path(summary['curve_path']).is_file())

    def test_no_valid_checkpoint_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = LinearSGDWrapper(lr=1e38, max_epochs=2)
            result = reg.fit(self.xt, self.yt, self.xv, self.yv, tmp, "bad")
            self.assertEqual(result['status'], 'diverged')
            self.assertIsNone(result['checkpoint_path'])
            with self.assertRaises(RuntimeError):
                reg.predict(self.xv)
            self.xt[0, 0] = np.nan
            with self.assertRaisesRegex(AssertionError, 'X_train'):
                reg.fit(self.xt, self.yt, self.xv, self.yv, tmp, "nan")

    def test_legacy_model_fit(self):
        reg = get_model_spec('ridge').builder()
        _fit_model('ridge', reg, self.xt, self.yt, self.xv, self.yv)
        self.assertEqual(reg.predict(self.xv).shape, (20,))

    def test_runner_keeps_test_locked_and_refuses_overwrite(self):
        # Test rows intentionally have no Price: tuning must never read their labels.
        rows = [dict(Address=f'listing {i}', Price=str(y), features=self.xt[i].tolist()) for i, y in enumerate(self.yt)]
        rows += [dict(Address=f'validation {i}', Price=str(y), features=self.xv[i].tolist()) for i, y in enumerate(self.yv)]
        rows += [dict(Address=f'locked {i}') for i in range(10)]
        recipe = CountingRecipe()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(approach_runner, 'PROJECT_ROOT', Path(tmp)), \
                patch.object(approach_runner, 'load_raw_records', return_value=rows), \
                patch.object(approach_runner, 'load_frozen_splits', return_value=(
                    np.arange(100), np.arange(100, 120), np.arange(120, 130))), \
                patch.object(approach_runner, 'build_feature_recipe', return_value=recipe):
            kwargs = dict(author='test', features='P4', target='raw_price',
                          params=dict(max_epochs=2), cpu_config=None, eval_test=False,
                          seed=42, notes='', csv_path=None, splits_path=EXPERIMENTS_SPLITS_PATH,
                          run_id='locked')
            result = approach_runner.run_linear_sgd(**kwargs)
            self.assertIsNone(result['metrics']['test'])
            self.assertEqual(recipe.transform_calls, 1)
            self.assertEqual(result['scaler_n_samples_seen'], 100)
            self.assertTrue(Path(result['saved_path']).is_file())
            shared_path = Path(result['shared_saved_path'])
            shared = json.loads(shared_path.read_text(encoding='utf-8'))
            self.assertEqual(shared['task'], 'regression')
            self.assertEqual(shared['metrics'], result['metrics'])
            self.assertEqual(shared['target']['unit'], 'billion_vnd')
            self.assertIn('locked', (Path(tmp) / 'experiments' / 'leaderboard.md').read_text(encoding='utf-8'))
            before = shared_path.read_bytes()
            sync_existing_run(result['saved_path'], Path(tmp))
            self.assertEqual(shared_path.read_bytes(), before)
            self.assertFalse(run_lock_path(Path(tmp), 'locked').exists())
            with self.assertRaises(FileExistsError):
                approach_runner.run_linear_sgd(**kwargs)

    def test_run_lock_excludes_another_process_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = run_lock_path(Path(tmp), 'concurrent')
            code = ('import sys; from src.experiment_io import file_lock; '
                    '\nwith file_lock(sys.argv[1]):\n print("reserved", flush=True)\n sys.stdin.readline()')
            owner = subprocess.Popen([sys.executable, '-c', code, str(lock)], cwd=PROJECT_ROOT,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(owner.stdout.readline().strip(), 'reserved')
                with patch.object(approach_runner, 'PROJECT_ROOT', Path(tmp)), \
                        patch.object(approach_runner, 'load_raw_records') as load:
                    with self.assertRaisesRegex(FileExistsError, 'reserved'):
                        approach_runner.run_linear_sgd(
                            author='test', features='P4', target='raw', params={}, cpu_config={},
                            eval_test=False, seed=42, notes='', csv_path=None,
                            splits_path=EXPERIMENTS_SPLITS_PATH, run_id='concurrent')
                    load.assert_not_called()
                self.assertTrue(lock.exists())  # Rejected contender must not delete owner's lock.
            finally:
                owner.communicate(input='release\n', timeout=30)
            self.assertEqual(owner.returncode, 0)
            self.assertFalse(lock.exists())
            with self.assertRaises(RuntimeError):
                with file_lock(lock):
                    raise RuntimeError('failed training')
            with file_lock(lock):
                self.assertTrue(lock.exists())

    def test_shared_export_rejects_collisions_and_handles_no_checkpoint(self):
        import copy
        source = dict(run_id='failed', status='diverged', approach='approach_1_linear_sgd',
                      author='test', target=dict(name='raw', unit='billion_vnd'),
                      features=dict(name='P4_geo_priors', n_features=345),
                      model=dict(name='linear_sgd', params={}),
                      metrics=dict(val=None, test=None), val_metrics=None, train_seconds=1.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'source.json'
            path.write_text(json.dumps(source), encoding='utf-8')
            shared = sync_existing_run(path, root)
            baseline = copy.deepcopy(source)
            baseline.update(run_id='baseline', status='completed')
            baseline['metrics']['val'] = dict(rmse=2., mae=1., r2=0., mape=10., medae=1.,
                                             within_10pct=.1, within_20pct=.2, within_1ty=.3)
            (shared.parent / 'baseline.json').write_text(json.dumps(baseline), encoding='utf-8')
            md = build_leaderboard(shared.parent, root / 'leaderboard.md', root / 'leaderboard.csv')
            self.assertIn('not ranked', md)
            self.assertIn('`failed`', md)
            self.assertIn('`baseline`', md)
            with (root / 'leaderboard.csv').open(newline='', encoding='utf-8') as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 1)
            self.assertNotIn(b'\r\r\n', (root / 'leaderboard.csv').read_bytes())
            before = shared.read_bytes()
            source['notes'] = 'different result with same ID'
            path.write_text(json.dumps(source), encoding='utf-8')
            with self.assertRaises(FileExistsError):
                sync_existing_run(path, root)
            self.assertEqual(shared.read_bytes(), before)

    def test_full_pipeline_roundtrip_in_fresh_process(self):
        rows = [dict(zip(RAW_NUM_COLS, [str(x) for x in (*row, 2., 3.)])) for row in self.xt]
        val_rows = [dict(zip(RAW_NUM_COLS, [str(x) for x in (*row, 2., 3.)])) for row in self.xv]
        recipe = build_feature_recipe('P0_numeric', scale_numeric=False, preserve_nans=False)
        xt, xv = recipe.fit_transform(rows, self.yt), recipe.transform(val_rows)
        with tempfile.TemporaryDirectory() as tmp:
            reg = LinearSGDWrapper(max_epochs=3)
            reg.fit(xt, self.yt, xv, self.yv, tmp, 'roundtrip')
            bundle = Path(tmp) / 'pipeline.joblib'
            LinearSGDPipeline(recipe, reg).save(bundle)
            expected = reg.predict(xv)
            val_file = Path(tmp) / 'rows.json'
            val_file.write_text(json.dumps([dict(row, Price='99999') for row in val_rows]), encoding='utf-8')
            code = ('import json, sys; from src.approach_1_linear_sgd.pipeline import LinearSGDPipeline; '
                    'from pathlib import Path; p=LinearSGDPipeline.load(sys.argv[1]); '
                    'print(json.dumps(p.predict(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))).tolist()))')
            # Remove standalone checkpoint: inference must depend only on the bundle.
            (Path(tmp) / 'checkpoints' / 'roundtrip_best.pt').unlink()
            child = subprocess.run([sys.executable, '-c', code, str(bundle), str(val_file)],
                                   cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=60)
            np.testing.assert_allclose(json.loads(child.stdout), expected, rtol=0, atol=1e-6)

    def test_interrupted_training_preserves_completed_epochs(self):
        original = torch.optim.SGD.step
        calls = 0
        def interrupt(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return original(optimizer, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.optim.SGD, 'step', interrupt):
            reg = LinearSGDWrapper(max_epochs=5)
            with self.assertRaises(KeyboardInterrupt):
                reg.fit(self.xt, self.yt, self.xv, self.yv, tmp, 'interrupted')
            with (Path(tmp) / 'curves' / 'interrupted_history.csv').open(newline='') as fh:
                history = list(csv.DictReader(fh))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]['epoch'], '1')
            self.assertTrue((Path(tmp) / 'checkpoints' / 'interrupted_best.pt').is_file())

    def test_cli_exit_status_even_with_best_checkpoint(self):
        from scripts.run_experiment import main
        metrics = dict(rmse=1., mae=1., r2=0., mape=10., within_20pct=.5)
        for status, expected_code in [('completed', 0), ('diverged', 1)]:
            result = dict(run_id='probe', status=status, metrics=dict(val=metrics, test=None),
                          features=dict(n_features=4), train_seconds=1., hardware='CPU')
            with patch('sys.argv', ['run_experiment.py', '--features', 'P4', '--model', 'linear_sgd']), \
                    patch('scripts.run_experiment.run_and_log', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), expected_code)
            self.assertIn(status, output.getvalue())

    def test_cli_preserves_legacy_config_but_requires_test_flag_for_linear_sgd(self):
        from scripts.run_experiment import main
        metrics = dict(rmse=1., mae=1., r2=0., mape=10., within_20pct=.5)
        result = dict(run_id='probe', metrics=dict(val=metrics, test=None),
                      features=dict(n_features=4), train_seconds=1., hardware='CPU')
        for model, flags, expected in [('ridge', [], True), ('linear_sgd', [], False),
                                        ('linear_sgd', ['--eval-test'], True)]:
            cfg = dict(model=model, features='P4_geo_priors', eval_test=True)
            with patch('sys.argv', ['run_experiment.py', '--config', 'example.json', *flags]), \
                    patch('scripts.run_experiment._load_config_file', return_value=cfg), \
                    patch('scripts.run_experiment.run_and_log', return_value=result) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
                self.assertEqual(run.call_args.kwargs['eval_test'], expected)


if __name__ == '__main__':
    unittest.main()
