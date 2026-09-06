class dotdict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        data = super().copy()
        return self.__class__(data)


def get_iter_file(iteration: int):
    return f'iteration-{iteration:04d}.pkl'


def scale_temp(scale_factor: float, min_temp: float, cur_temp: float, turns: int, const_max_turns: int) -> float:
    if const_max_turns and (turns + 1) % int(scale_factor * const_max_turns) == 0:
        return max(min_temp, cur_temp / 2)
    else:
        return cur_temp


def default_temp_scaling(*args, **kwargs) -> float:
    return scale_temp(0.15, 0.2, *args, **kwargs)


def const_temp_scaling(temp, *args, **kwargs) -> float:
    return temp


def dotdict_to_dict(data):
    if isinstance(data, dotdict):
        return {key: dotdict_to_dict(value) for key, value in data.items()}
    if isinstance(data, list):
        return [dotdict_to_dict(value) for value in data]
    return data


def plot_mcts_tree(*args, **kwargs):
    try:
        from AlphaZeroGUI.CustomGUI import MCTSTreeDialog
        MCTSTreeDialog(*args, **kwargs).exec_()
    except ImportError:
        print('Could not import AlphaZeroGUI. MCTS tree cannot be displayed.')
