# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""``python -m cost_core`` runs the ``ce-core`` command.

For machines where pip's scripts folder is not on the PATH, which is common on
managed Windows installs: ``python -m cost_core demo evm`` works wherever the
package imports.
"""

from cost_core.cli import main

main()
