import pathlib
from private_modules import load_yaml_config, strpath2path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar, Optional, Dict

class ProjConfig(BaseSettings):
    eps: float = 1e-7
    base_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
    base_config_f: pathlib.Path = base_dir.joinpath('configs/base.yml')
    base_config: Dict = load_yaml_config(base_config_f)
    proj_db_dir: pathlib.Path = base_dir.joinpath('ProjDB')
    DATABASE_dir: pathlib.Path = strpath2path(base_config['data']['database_dir'])

    # --- WEST boundary data subdirs (derived from DATABASE_dir) ---
    dcs_org_dir: ClassVar[pathlib.Path] = DATABASE_dir / 'DCSOrg'
    dcs_h5_dir: ClassVar[pathlib.Path] = DATABASE_dir / 'DCSH5'
    data_org_dir: ClassVar[pathlib.Path] = DATABASE_dir / 'DataOrg'
    merged_dir: ClassVar[pathlib.Path] = DATABASE_dir / 'Merged'

    # --- ProjDB output layout (derived from proj_db_dir) ---
    stats_dir: ClassVar[pathlib.Path] = proj_db_dir / 'Stats'
    figures_dir: ClassVar[pathlib.Path] = proj_db_dir / 'Figures'
    shot_status_csv: ClassVar[pathlib.Path] = stats_dir / 'shot_status.csv'
    shot_freq_stats_csv: ClassVar[pathlib.Path] = stats_dir / 'shot_freq_stats.csv'
    lcfs_fig_dir: ClassVar[pathlib.Path] = stats_dir / 'LCFS'
    sig_freqs_fig_dir: ClassVar[pathlib.Path] = stats_dir / 'SigFreqs'
    dcs_actual_vs_ref_dir: ClassVar[pathlib.Path] = figures_dir / 'dcs_actual_vs_ref'

    # --- DCS scope column semantics: (label, matrix-column-index) ---
    dcs_columns: ClassVar[tuple] = (('ref', 0), ('actual', 3))
    # h5 key of the reconstructed-boundary target signal
    gmag_bnd_key: ClassVar[str] = 'targets/GMAG_BND'

    # --- tunable thresholds (overridable via PROJ_* env vars) ---
    dcs_trunc_min: int = 1000       # DCS records shorter than this = "truncated"
    downsample_tol: float = 0.9     # < this * nominal => "downsampled"

    model_config = SettingsConfigDict(
        env_file=('.env', '.env.local'),
        env_prefix='PROJ_',
    )    
__proj_config__: Optional[ProjConfig] = None

def get_proj_config() -> ProjConfig:
    global __proj_config__
    if __proj_config__ is None:
        __proj_config__ = ProjConfig()
    return __proj_config__