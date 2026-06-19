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

    # --- DCS scope column semantics: (label, matrix-column-index) ---
    dcs_columns: ClassVar[tuple] = (('ref', 0), ('actual', 3))
    # h5 key of the reconstructed-boundary target signal
    gmag_bnd_key: ClassVar[str] = 'targets/GMAG_BND'

    # --- tunable thresholds (overridable via PROJ_* env vars) ---
    dcs_trunc_min: int = 1000       # DCS records shorter than this = "truncated"
    downsample_tol: float = 0.9     # < this * nominal => "downsampled"
    flat_top_min_s: float = 3.0     # keep shots whose total Ip flat-top exceeds this

    # --- WEST boundary data subdirs (derived from DATABASE_dir) ---
    @property
    def dcs_org_dir(self) -> pathlib.Path:
        return self.DATABASE_dir / "DCSOrg"

    @property
    def dcs_h5_dir(self) -> pathlib.Path:
        return self.DATABASE_dir / "DCSH5"

    @property
    def data_org_dir(self) -> pathlib.Path:
        return self.DATABASE_dir / "DataOrg"

    @property
    def merged_dir(self) -> pathlib.Path:
        return self.DATABASE_dir / "Merged"

    @property
    def npz_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "Npz"

    # --- IMAS-native LCFS sweep layout ---
    @property
    def imas_h5_dir(self) -> pathlib.Path:
        # IMAS/<shot>.h5 sits next to Boundary/: <DATABASE_PATH>/DataBase/WEST/IMAS
        return self.DATABASE_dir.parent / "IMAS"

    @property
    def imas_npz_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "Npz" / "imas"

    @property
    def imas_sweep_stats_dir(self) -> pathlib.Path:
        return self.stats_dir / "imas_input_sweep"

    # --- data output layout (CSV tables under ProjDB/Stats) ---
    @property
    def stats_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "Stats"

    # --- model output layout (training runs / hyperparameter tuning under ProjDB) ---
    @property
    def trains_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "trains"

    @property
    def tunes_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "tunes"

    @property
    def inferences_dir(self) -> pathlib.Path:
        return self.proj_db_dir / "inferences"

    @property
    def shot_status_csv(self) -> pathlib.Path:
        return self.stats_dir / "shot_status.csv"

    @property
    def shot_freq_stats_csv(self) -> pathlib.Path:
        return self.stats_dir / "shot_freq_stats.csv"

    @property
    def signal_freq_summary_csv(self) -> pathlib.Path:
        return self.stats_dir / "signal_freq_summary.csv"

    # --- figure output layout (generated PNGs under <project root>/figs) ---
    @property
    def figs_dir(self) -> pathlib.Path:
        return self.base_dir / "figs"

    @property
    def lcfs_fig_dir(self) -> pathlib.Path:
        return self.figs_dir / "LCFS"

    @property
    def sig_freqs_fig_dir(self) -> pathlib.Path:
        return self.figs_dir / "SigFreqs"

    @property
    def dcs_actual_vs_ref_dir(self) -> pathlib.Path:
        return self.figs_dir / "dcs_actual_vs_ref"

    def data_paths(self) -> Dict[str, pathlib.Path]:
        """Common database paths used across the data-prep pipeline."""
        return {
            "bnd_dir": self.data_org_dir,
            "dcs_dir": self.dcs_h5_dir,
            "stats_dir": self.stats_dir,
        }

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