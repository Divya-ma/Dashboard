"""Every callback module of the dashboard, registered in one place (ui/app.py and the wiring tests both call register_all)."""

from ui.callbacks.archive_callbacks import register_archive_callbacks
from ui.callbacks.churn_callbacks import register_churn_callbacks
from ui.callbacks.corr_matrix_callbacks import register_corr_matrix_callbacks
from ui.callbacks.correlation_callbacks import register_correlation_callbacks
from ui.callbacks.curve_callbacks import register_curve_callbacks
from ui.callbacks.curve_compare_callbacks import register_curve_compare_callbacks
from ui.callbacks.dissolve_callbacks import register_dissolve_callbacks
from ui.callbacks.exposure_callbacks import register_exposure_callbacks
from ui.callbacks.home_callbacks import register_home_callbacks
from ui.callbacks.oi_callbacks import register_oi_callbacks
from ui.callbacks.rollover_callbacks import register_rollover_callbacks
from ui.callbacks.rollover_study_callbacks import register_rollover_study_callbacks
from ui.callbacks.seasonality_callbacks import register_seasonality_callbacks
from ui.callbacks.settings_callbacks import register_settings_callbacks
from ui.callbacks.shell_callbacks import register_callbacks as register_shell_callbacks
from ui.callbacks.structure_builder_callbacks import register_structure_builder_callbacks
from ui.callbacks.structure_detail_callbacks import register_structure_detail_callbacks
from ui.callbacks.structures_callbacks import register_structures_callbacks
from ui.callbacks.var_callbacks import register_var_callbacks

REGISTRATIONS = (
    register_shell_callbacks, register_home_callbacks, register_settings_callbacks, register_structures_callbacks,
    register_structure_builder_callbacks, register_structure_detail_callbacks, register_dissolve_callbacks,
    register_correlation_callbacks, register_corr_matrix_callbacks, register_exposure_callbacks, register_var_callbacks,
    register_curve_callbacks, register_curve_compare_callbacks, register_oi_callbacks, register_churn_callbacks,
    register_seasonality_callbacks, register_rollover_callbacks, register_rollover_study_callbacks, register_archive_callbacks,
)


def register_all(app) -> None:
    for register in REGISTRATIONS:
        register(app)
