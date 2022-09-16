#!/usr/bin/env python# -*- encoding: utf-8 -*-"""@File    :   battery_data_eda.py    @Modify Time      @Author    @Version    @Description------------      -------    --------    -----------2022/9/16 17:15   SeafyLiang   1.0        电池数据探索性分析"""# Databricks notebook sourceimport matplotlib.pyplot as pltfrom dataset_util import *# 加载原始数据B0005 = loadMat('B0005.mat')B0006 = loadMat('B0006.mat')B0007 = loadMat('B0007.mat')B0018 = loadMat('B0018.mat')# 获取容量数据B0005_capacity = getBatteryCapacity(B0005)B0006_capacity = getBatteryCapacity(B0006)B0007_capacity = getBatteryCapacity(B0007)B0018_capacity = getBatteryCapacity(B0018)# 1、循环与容量图fig, ax = plt.subplots(1, figsize=(12, 8))ax.plot(B0005_capacity[0], B0005_capacity[1], color='red', label='B0005')ax.plot(B0006_capacity[0], B0006_capacity[1], color='purple', label='B0006')ax.plot(B0007_capacity[0], B0007_capacity[1], color='orangered', label='B0007')ax.plot(B0018_capacity[0], B0018_capacity[1], color='green', label='B0018')ax.set(xlabel='Cycles', ylabel='Capacity', title='Capacity degradation | type=discharge,T=24°C' + '\n')ax.set_facecolor('beige')plt.legend()plt.show()# 获取充电数据B0005_charging = getChargingValues(B0005, 0)B0006_charging = getChargingValues(B0006, 0)B0007_charging = getChargingValues(B0007, 0)B0018_charging = getChargingValues(B0018, 0)# 2、充电状态下电压、电流、温度时序图charging_labels = ['Voltage_measured', 'Current_measured', 'Temperature_measured']indx = 1for label in charging_labels:    fig2 = plt.figure(dpi=200, figsize=(20, 7))    plt.title('Charging performance-%s' % str(label))    plt.xlabel('Time(s)')    plt.ylabel(label)    # 隐藏x，y轴刻度尺    plt.xticks(alpha=0)    plt.yticks(alpha=0)    ax1 = fig2.add_subplot(4, 1, 1)    ax2 = fig2.add_subplot(4, 1, 2)    ax3 = fig2.add_subplot(4, 1, 3)    ax4 = fig2.add_subplot(4, 1, 4)    ax1.plot(B0005_charging[5], B0005_charging[indx], color='red', label="B0005")    ax1.legend()    ax2.plot(B0006_charging[5], B0006_charging[indx], color='orange', label="B0006")    ax2.legend()    ax3.plot(B0007_charging[5], B0007_charging[indx], color='blue', label="B0007")    ax3.legend()    ax4.plot(B0018_charging[5], B0018_charging[indx], color='pink', label="B0018")    ax4.legend()    indx += 1    plt.show()# 获取放电数据B0005_discharging = getDischargingValues(B0005, 1)B0006_discharging = getDischargingValues(B0006, 1)B0007_discharging = getDischargingValues(B0007, 1)# B0018_discharging = getDischargingValues(B0018, 1)  # 报错，没有电压测量值# 3、放电状态下电压、电流、温度时序图indx = 1for label in charging_labels:    fig3 = plt.figure(dpi=200, figsize=(20, 7))    plt.title('DisCharging performance-%s' % str(label))    plt.xlabel('Time(s)')    plt.ylabel(label)    # 隐藏x，y轴刻度尺    plt.xticks(alpha=0)    plt.yticks(alpha=0)    ax1 = fig3.add_subplot(3, 1, 1)    ax2 = fig3.add_subplot(3, 1, 2)    ax3 = fig3.add_subplot(3, 1, 3)    ax1.plot(B0005_discharging[5], B0005_discharging[indx], color='red', label="B0005")    ax1.legend()    ax2.plot(B0006_discharging[5], B0006_discharging[indx], color='orange', label="B0006")    ax2.legend()    ax3.plot(B0007_discharging[5], B0007_discharging[indx], color='blue', label="B0007")    ax3.legend()    indx += 1    plt.show()