import warnings
import numpy as np
from pprint import pprint
from sympy import Symbol, Function, Eq, Derivative, lambdify, Expr
from sympy.core.relational import Equality
from sympy.abc import t
from typing import List, Dict, TypedDict, NotRequired, AnyStr
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from pathlib import Path
import json

class EquationDescription(TypedDict):
    equations:List[Equality]
    constraints: NotRequired[List[Equality]]
    symbols:NotRequired[List[Symbol]]
    base_equations: NotRequired[List[Equality]]

'''
TODO:   Think about time dilation, as a lot of the signalling phenomena happen on timescale
        invisible from frame point of view (50ms pulse of light looks identical to every other if
        our time resolution is 1 frame (1 second)).
        On the second thought, that should probably happen outside of this function,
        maybe on the level of preparing the dataframe? Just a helper func or sth?
'''

class Model:
    fit_result = None

    def __init__(self, name, states, parameters, model_definition, t_dep,
                 t_func=None, ivp_method='LSODA', minimizer_method='L-BFGS-B',
                 group_to_light=None):
        '''
        We give it a name, we give it all the symbolic equations, and params (defaults?)

        Pipeline:
            data ---------------------------------------.
                                                        |
            eqs + params + light_fn -> numeric system  -L--> make loss function -> feed to minimizer

        What's a good API to insert EQs? Do i need to input symbol for t?
        '''
        if (t_func is None) == (group_to_light is None):
            raise ValueError("Provide exactly one of t_func or group_to_light")

        self.name = name
        self.group_to_light = group_to_light
        self.states = states
        self.parameters = parameters
        self.model_definition_f = model_definition
        model = model_definition(parameters, states)
        self.eqs = model['equations']
        self.symbols = model['symbols']
        self.base_equations = model['base_equations']
        if not isinstance(t_dep, str) or t_dep not in self.symbols:
            raise ValueError('time-dependant variable must exist within equations')
        self.ivp_method = ivp_method
        self.minimizer_method = minimizer_method
        self.t_func = t_func
        self.t_dep = t_dep

    def _get_t_func(self, group_id):
        if self.group_to_light is None:
            return self.t_func
        return self.group_to_light[group_id]

    def _build_param_vector(self, fitted_params):
        vals = []
        for p in self.parameters:
            if p in fitted_params:
                vals.append(fitted_params[p])
            else:
                warnings.warn(f"Parameter '{p}' not in fitted params, using default 1.0")
                vals.append(1.0)
        return np.array(vals, dtype=np.float64)

    def _simulate_group(self, system, p_full, t_func, times, y0, t_args):
        sol = solve_ivp(
            lambda t, y: system(t, y, p_full, t_func, t_args),
            [times[0], times[-1]], y0,
            t_eval=times, method=self.ivp_method, rtol=1e-8
        )
        return sol

    @staticmethod
    def _group_t_args(group_data, base_t_args):
        meta_cols = [col for col in group_data.columns if col not in {'time', 'y'}]
        group_meta = {col: group_data.iloc[0][col] for col in meta_cols}
        return {**base_t_args, **group_meta}

    def make_numerical(self, module='numpy'):
        free_vars = set()
        for eq in self.eqs:
            free_vars.update(eq.rhs.free_symbols)

        # cannot just use self.states cause transform() could have reduced them
        active_states = [s for s in self.states if self.symbols[s] in free_vars]
        self.active_states = active_states
        states = [self.symbols[s] for s in active_states]
        parameters = [self.symbols[p] for p in self.parameters]

        arg_list = (t, *states, *parameters, self.symbols[self.t_dep])
        numerical_funcs = [lambdify(arg_list, eq.rhs, modules=module)
                           for eq in self.eqs]

        def system(t, y, params, t_func, t_args=None):
            y = np.maximum(y, 1e-3)
            res = t_func(t, t_args)
            args = [t, *y, *params, res]
            return [f(*args) for f in numerical_funcs]

        return system

    def simulate(self, times, y0, group=None, t_args=None):
        assert self.fit_result is not None
        assert group is not None
        if t_args is None:
            t_args = {'group': group}
        t_func = self._get_t_func(group)
        p_full = self._build_param_vector(self.fit_result['fitted_params'])

        y0 = np.asarray(y0, dtype=np.float64)
        system = self.make_numerical()
        sol = self._simulate_group(system, p_full, t_func, times, y0, t_args)
        if not sol.success:
            warnings.warn('simulate: ODE solver failed')
        return sol

    def fit(self, dataframe, y0, parameters, t_args=None):
        """
        Fit model parameters to experimental data.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            Must contain columns:
            - 'time': float, time points in seconds
            - 'y': float, observed values (e.g., KTR levels)
            - 'group': str/int, experiment/cell identifier for grouping
        y0 : array_like
            Initial conditions for the ODE system
        parameters : dict or str or PosixPath
            If dict: {param_name: initial_value} for parameters to optimize
            If str or PosixPath: path to JSON config file with parameter defaults
        t_args : dict
            Arguments to pass to the time-dependent function (e.g., light intensity)
        """
        y0 = np.asarray(y0, dtype=np.float64)

        # Handle parameters input - could be dict, str, or PosixPath
        config_path = None
        if isinstance(parameters, (str, Path)):
            config_path = Path(parameters)
            parameters = self.read_config(config_path)
        elif not isinstance(parameters, dict):
            raise ValueError("parameters must be dict, str, or PosixPath")

        if t_args is None:
            t_args = {}
        elif not isinstance(t_args, dict):
            raise ValueError("t_args must be a dict")

        base_t_args = dict(t_args)

        required_cols = {'time', 'y', 'group'}
        if not required_cols.issubset(dataframe.columns):
            missing = required_cols - set(dataframe.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.parameters_to_fit = parameters
        param_names = list(parameters.keys())

        system = self.make_numerical()

        def objective_log(p_fit_log_values):
            return objective_normal(np.exp(p_fit_log_values))

        def objective_normal(p_fit_values):
            fitted = dict(zip(param_names, p_fit_values))
            p_full = self._build_param_vector(fitted)

            total_loss = 0.0
            for group_id in dataframe['group'].unique():
                group_data = dataframe[dataframe['group'] == group_id].sort_values('time')
                times = group_data['time'].values.astype(np.float64, copy=False)
                observed_data = group_data['y'].values.astype(np.float64, copy=False)
                t_func = self._get_t_func(group_id)
                current_t_args = self._group_t_args(group_data, base_t_args)

                try:
                    sol = self._simulate_group(system, p_full, t_func, times, y0, current_t_args)
                    if sol.success:
                        loss = np.sum((sol.y[-1] - observed_data)**2)
                        total_loss += loss
                    else:
                        return 1e10
                except Exception as e:
                    warnings.warn(f"Error solving ODE for group {group_id}: {e}")
                    return 1e10

            return total_loss

        p_init = [parameters[name] for name in param_names]

        use_log_transform = self.minimizer_method in ['L-BFGS-B', 'trust-constr', 'SLSQP']

        if use_log_transform:
            p_init_log = np.log(np.maximum(p_init, 1e-10))
            objective_func = objective_log
            bounds = [(-10, 5)] * len(p_init)  # exp(-10) to exp(5) ≈ 4.5e-5 to 148
        else:
            objective_func = objective_normal
            bounds = [(1e-6, 100)] * len(p_init)

        if self.minimizer_method == 'L-BFGS-B':
            options = {'maxiter': 10000, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 50}
        elif self.minimizer_method == 'trust-constr':
            options = {'maxiter': 10000, 'xtol': 1e-8, 'gtol': 1e-6}
        else:
            options = {'maxiter': 10000}

        initial_params = p_init_log if use_log_transform else p_init

        result = minimize(
            objective_func, initial_params,
            method=self.minimizer_method,
            bounds=bounds,
            options=options
        )

        final_params = np.exp(result.x) if use_log_transform else result.x
        fitted_params = dict(zip(param_names, final_params))

        # Per-group statistics
        n_data_points = len(dataframe)
        n_fitted_params = len(param_names)
        p_full = self._build_param_vector(fitted_params)
        group_stats = {}

        for group_id in dataframe['group'].unique():
            group_data = dataframe[dataframe['group'] == group_id].sort_values('time')
            times = group_data['time'].values
            observed_data = group_data['y'].values
            t_func = self._get_t_func(group_id)
            current_t_args = self._group_t_args(group_data, base_t_args)

            try:
                sol = self._simulate_group(system, p_full, t_func, times, y0, current_t_args)
                if sol.success:
                    predicted_data = sol.y[-1]
                    residuals = predicted_data - observed_data
                    mse = np.mean(residuals**2)
                    r_squared = 1 - np.sum(residuals**2) / np.sum((observed_data - np.mean(observed_data))**2)
                    group_stats[str(group_id)] = {
                        'mse': float(mse),
                        'rmse': float(np.sqrt(mse)),
                        'r_squared': float(r_squared),
                        'n_points': len(times),
                        'max_observed': float(np.max(observed_data)),
                        'max_predicted': float(np.max(predicted_data))
                    }
            except Exception:
                group_stats[str(group_id)] = {'error': 'simulation_failed'}

        overall_mse = result.fun / n_data_points
        aic = 2 * n_fitted_params + n_data_points * np.log(result.fun / n_data_points) if result.fun > 0 else np.inf
        bic = np.log(n_data_points) * n_fitted_params + n_data_points * np.log(result.fun / n_data_points) if result.fun > 0 else np.inf

        fit_result_data = {
            'fitted_params': fitted_params,
            'loss': result.fun,
            'success': result.success,
            'message': result.message,
            'n_experiments': len(dataframe['group'].unique()),
            'optimization_info': {
                'method': self.minimizer_method,
                'n_iterations': getattr(result, 'nit', None),
                'n_function_evaluations': getattr(result, 'nfev', None),
                'log_transform_used': use_log_transform,
                'final_gradient_norm': float(np.linalg.norm(result.jac)) if getattr(result, 'jac', None) is not None else None
            },
            'fit_statistics': {
                'overall_mse': float(overall_mse),
                'rmse': float(np.sqrt(overall_mse)),
                'aic': float(aic),
                'bic': float(bic),
                'degrees_of_freedom': n_data_points - n_fitted_params,
                'n_data_points': n_data_points,
                'n_fitted_params': n_fitted_params,
                'group_statistics': group_stats
            },
            'params': {
                "fitted": parameters,
                "all": self.parameters
            }
        }

        if config_path is not None:
            fit_result_data['config_path'] = str(config_path)

        self.fit_result = fit_result_data
        return self.fit_result

    def read_config(self, config_path):
        """
        Read parameter configuration from JSON file.

        Handles different JSON structures:
        1. Simple dict: {param_name: value, ...}
        2. Result structure with 'fitted_params': {fitted_params: {...}}
        3. Result structure with 'params': {params: {...}}
        4. Complex nested structures
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            data = json.load(f)

        params_dict = None

        if self._is_simple_param_dict(data):
            params_dict = data
        elif 'fitted_params' in data and isinstance(data['fitted_params'], dict):
            params_dict = data['fitted_params']
        elif 'params' in data and isinstance(data['params'], dict):
            if 'to_fit' in data['params'] and isinstance(data['params']['to_fit'], dict):
                params_dict = data['params']['to_fit']
            else:
                params_dict = data['params']
        elif 'meta' in data and isinstance(data['meta'], dict) and 'fitted_params' in data['meta']:
            params_dict = data['meta']['fitted_params']
        else:
            params_dict = self._extract_params_from_nested(data)

        if params_dict is None:
            raise ValueError(f"Could not extract parameters from {config_path}")

        filtered_params = {param: params_dict[param]
                          for param in self.parameters
                          if param in params_dict}

        if not filtered_params:
            raise ValueError(f"No matching parameters found in config. "
                           f"Config contains: {list(params_dict.keys())}, "
                           f"Model expects: {self.parameters}")

        print(f"Loaded {len(filtered_params)} parameters from {config_path}")
        missing_params = set(self.parameters) - set(filtered_params.keys())
        if missing_params:
            print(f"Missing parameters (will use defaults): {missing_params}")

        return filtered_params

    def _is_simple_param_dict(self, data):
        if not isinstance(data, dict):
            return False
        metadata_keys = {'fitted_params', 'params', 'meta', 'loss', 'success',
                        'message', 'optimization_info', 'fit_statistics'}
        has_metadata = any(key in data for key in metadata_keys)
        all_numeric = all(isinstance(v, (int, float)) for v in data.values())
        return not has_metadata and all_numeric

    def _extract_params_from_nested(self, data):
        if not isinstance(data, dict):
            return None

        def search_for_params(obj, path=""):
            if not isinstance(obj, dict):
                return None
            matches = sum(1 for param in self.parameters if param in obj)
            if matches > 0:
                return obj, matches, path
            best = None
            for key, value in obj.items():
                result = search_for_params(value, f"{path}.{key}" if path else key)
                if result and (best is None or result[1] > best[1]):
                    best = result
            return best

        result = search_for_params(data)
        if result:
            params_dict, match_count, path = result
            print(f"Found {match_count} matching parameters at path: {path}")
            return params_dict
        return None

    def save_results(self, path):
        assert self.fit_result is not None, "To save results you gotta have results."
        path = Path(path)
        pprint(self.fit_result)
        with open(path, 'w') as f:
            json.dump(self.fit_result, f, indent=2)


if __name__ == "__main__":
    import pandas as pd
    from sympy import symbols, Function

    # --- Define a simple 1-state ODE: dx/dt = -k * x + L(t) ---
    x_sym, k_sym, L_sym = symbols('x k L')
    x_func = Function('x')(t)

    def simple_model_def(parameters, states):
        eq = Eq(x_func.diff(t), -k_sym * x_sym + L_sym)
        return {
            'equations': [eq],
            'symbols': {'x': x_sym, 'k': k_sym, 'L': L_sym},
            'base_equations': [eq],
        }

    def light_fn(t_val, t_args=None):
        return 1.0 if t_val < 5 else 0.0

    # Test 1: construction
    m = Model(
        name='test_decay',
        states=['x'],
        parameters=['k'],
        model_definition=simple_model_def,
        t_dep='L',
        t_func=light_fn,
    )
    assert m.name == 'test_decay'
    assert m.t_dep == 'L'
    print("[PASS] construction")

    # Test 2: must provide exactly one of t_func / group_to_light
    try:
        Model('bad', ['x'], ['k'], simple_model_def, 'L')
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        Model('bad', ['x'], ['k'], simple_model_def, 'L',
              t_func=light_fn, group_to_light={'g': light_fn})
        assert False, "should have raised"
    except ValueError:
        pass
    print("[PASS] t_func / group_to_light validation")

    # Test 3: make_numerical produces a callable system
    system = m.make_numerical()
    dy = system(0.0, [0.0], [0.5], light_fn)
    assert len(dy) == 1
    assert abs(dy[0] - 1.0) < 1e-8  # dx/dt = -0.5*0 + 1 = 1
    dy2 = system(0.0, [2.0], [0.5], light_fn)
    assert abs(dy2[0] - 0.0) < 1e-8  # dx/dt = -0.5*2 + 1 = 0
    print("[PASS] make_numerical")

    # Test 4: _build_param_vector
    pv = m._build_param_vector({'k': 3.14})
    assert pv.shape == (1,) and abs(pv[0] - 3.14) < 1e-8
    print("[PASS] _build_param_vector")

    # Test 5: _simulate_group
    times = np.linspace(0, 10, 50)
    sol = m._simulate_group(system, np.array([0.5]), light_fn, times, np.array([0.0]), None)
    assert sol.success
    assert sol.y.shape[1] == 50
    print("[PASS] _simulate_group")

    # Test 6: fit on synthetic data
    # Generate data from known k=0.5
    true_k = 0.5
    true_sol = solve_ivp(
        lambda t_val, y: system(t_val, y, [true_k], light_fn),
        [0, 10], [0.0], t_eval=np.linspace(0, 10, 30), method='LSODA', rtol=1e-8
    )
    df = pd.DataFrame({
        'time': true_sol.t,
        'y': true_sol.y[0],
        'group': 'g1',
    })
    result = m.fit(df, y0=[0.0], parameters={'k': 1.0})
    recovered_k = result['fitted_params']['k']
    assert abs(recovered_k - true_k) < 0.05, f"expected ~{true_k}, got {recovered_k}"
    assert result['success']
    assert 'group_statistics' in result['fit_statistics']
    print(f"[PASS] fit (recovered k={recovered_k:.4f}, true={true_k})")

    # Test 7: simulate after fit
    sim_sol = m.simulate(np.linspace(0, 10, 20), y0=[0.0], group='g1')
    assert sim_sol.success
    assert sim_sol.y.shape == (1, 20)
    print("[PASS] simulate")

    # Test 8: _is_simple_param_dict
    assert m._is_simple_param_dict({'k': 1.0, 'a': 2.0})
    assert not m._is_simple_param_dict({'k': 1.0, 'loss': 0.5})
    assert not m._is_simple_param_dict({'k': {'nested': 1}})
    print("[PASS] _is_simple_param_dict")

    print("\nAll tests passed.")
