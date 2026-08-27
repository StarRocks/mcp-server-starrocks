# Copyright 2021-present StarRocks, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


class SecretResolutionError(RuntimeError):
    """Raised when secret lookup is configured but cannot resolve a password."""


@dataclass(frozen=True)
class KeychainLookupConfig:
    service: str
    account: str


def resolve_password(*, user: str, explicit_password: str, explicit_password_provided: bool) -> str:
    """
    Resolve a StarRocks password.

    Explicitly configured passwords always win. Otherwise, an optional password
    file is checked before falling back to macOS Keychain lookup.
    """
    if explicit_password_provided:
        return explicit_password

    password_file = os.getenv('STARROCKS_PASSWORD_FILE')
    if password_file:
        return read_password_from_file(password_file)

    lookup = get_keychain_lookup_config(user)
    if lookup is None:
        return explicit_password

    return read_password_from_macos_keychain(lookup)


def read_password_from_file(path: str) -> str:
    """Read a password from a UTF-8 text file without exposing its contents."""
    try:
        # newline='' disables universal-newline translation so an embedded '\r'
        # stays part of the password; only one trailing line ending is stripped.
        with open(path, 'r', encoding='utf-8', newline='') as password_file:
            return password_file.read().removesuffix('\n').removesuffix('\r')
    except (OSError, UnicodeError) as exc:
        raise SecretResolutionError(
            f"Unable to read StarRocks password from STARROCKS_PASSWORD_FILE '{path}': {exc}"
        ) from exc


def get_keychain_lookup_config(user: str) -> Optional[KeychainLookupConfig]:
    """Build Keychain lookup config from environment variables, if configured."""
    service = os.getenv('STARROCKS_PASSWORD_KEYCHAIN_SERVICE')
    if not service:
        return None

    account = os.getenv('STARROCKS_PASSWORD_KEYCHAIN_ACCOUNT') or user
    return KeychainLookupConfig(service=service, account=account)


def read_password_from_macos_keychain(lookup: KeychainLookupConfig) -> str:
    """Read a generic password from macOS Keychain using the `security` CLI."""
    if sys.platform != 'darwin':
        raise SecretResolutionError(
            "STARROCKS_PASSWORD_KEYCHAIN_SERVICE is only supported on macOS because it relies on the `security` CLI."
        )

    security_command = shutil.which('security')
    if security_command is None and os.path.exists('/usr/bin/security'):
        security_command = '/usr/bin/security'
    if security_command is None:
        raise SecretResolutionError(
            "macOS Keychain lookup is configured, but the `security` command is not available."
        )

    result = subprocess.run(
        [security_command, 'find-generic-password', '-a', lookup.account, '-s', lookup.service, '-w'],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        details = f" {stderr}" if stderr else ""
        raise SecretResolutionError(
            f"Unable to read StarRocks password from macOS Keychain for service '{lookup.service}' "
            f"and account '{lookup.account}'. Create the item with `security add-generic-password` "
            f"or set STARROCKS_PASSWORD/STARROCKS_URL instead.{details}"
        )

    return result.stdout.removesuffix('\n')
