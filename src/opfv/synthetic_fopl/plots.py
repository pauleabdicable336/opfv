# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
#
# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

"""Optional matplotlib plots for synthetic F-OPL training curves."""

from __future__ import annotations

import matplotlib.pyplot as plt


def show_loss(opfv_opl, ips, dr, reg) -> None:
    num_iter = len(opfv_opl.train_loss)
    xticks = range(1, num_iter + 1)
    plt.style.use("ggplot")
    plt.plot(xticks, ips.train_loss, label="IPS")
    plt.plot(xticks, dr.train_loss, label="DR")
    plt.plot(xticks, reg.train_loss, label="Regression-based")
    plt.plot(xticks, opfv_opl.train_loss, label="OPFV")
    plt.xlabel("epochs")
    plt.xticks(list(xticks))
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.show()


def show_value(opfv_opl, ips, dr, reg) -> None:
    plt.style.use("ggplot")
    plt.plot(ips.test_value, label="IPS test")
    plt.plot(dr.test_value, label="DR test")
    plt.plot(reg.test_value, label="REG test")
    plt.plot(opfv_opl.test_value, label="OPFV test")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Test policy value")
    plt.legend()
    plt.show()
