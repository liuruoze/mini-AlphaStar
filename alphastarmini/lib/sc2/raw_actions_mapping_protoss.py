#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Raw actions mapping for protoss, purely built by human knowledge"

from pysc2.lib.actions import RAW_FUNCTIONS as F
from pysc2.lib.units import Protoss, Neutral

__author__ = "Ruo-Ze Liu"

SMALL_MAPPING = {
    "no_op": [None, None, 0],
    "Smart_pt": [None, None, 1],
    "raw_move_camera": [None, None, 0],
    "Train_Probe_quick": [Protoss.Nexus, None, 1],
    "Build_Pylon_pt": [Protoss.Probe, None, 1],
    "Build_Gateway_pt": [Protoss.Probe, None, 1],
    "Build_Assimilator_unit": [Protoss.Probe, Neutral.VespeneGeyser, 1],
    "Build_CyberneticsCore_pt": [Protoss.Probe, None, 1],
    "Train_Zealot_quick": [Protoss.Gateway, None, 8],
    "Train_Stalker_quick": [Protoss.Gateway, None, 8],
    "Harvest_Gather_unit": [Protoss.Probe, [Neutral.MineralField, Neutral.VespeneGeyser], 1],
    "Attack_pt": [[Protoss.Zealot, Protoss.Stalker], None, 24],
}

SMALL_LIST = [   
    F.Train_Probe_quick.id.value,
    F.Build_Pylon_pt.id.value,
    F.Build_Gateway_pt.id.value,
    # F.Build_Assimilator_unit.id.value,
    # F.Build_CyberneticsCore_pt.id.value,
    F.Train_Zealot_quick.id.value,
    # F.Train_Stalker_quick.id.value,
]

MEDIUM_LIST = [   
    F.Train_Probe_quick.id.value,
    F.Build_Nexus_pt.id.value,
    F.Build_Pylon_pt.id.value,
    F.Build_Gateway_pt.id.value,
    F.Build_Assimilator_unit.id.value,
    F.Build_CyberneticsCore_pt.id.value,
    F.Build_Forge_pt.id.value,
    F.Build_FleetBeacon_pt.id.value,
    F.Build_TwilightCouncil_pt.id.value,
    F.Build_PhotonCannon_pt.id.value,
    F.Build_Stargate_pt.id.value,
    F.Build_TemplarArchive_pt.id.value,
    F.Build_DarkShrine_pt.id.value,
    F.Build_RoboticsBay_pt.id.value,
    F.Build_RoboticsFacility_pt.id.value,
    F.Build_ShieldBattery_pt.id.value,
    F.Train_Zealot_quick.id.value,
    F.Train_Stalker_quick.id.value,
    F.Train_HighTemplar_quick.id.value,
    F.Train_DarkTemplar_quick.id.value,
    F.Train_Sentry_quick.id.value,
    F.Train_Adept_quick.id.value,
    F.Train_Phoenix_quick.id.value,
    # F.Train_Carrier_quick.id.value,
    F.Train_VoidRay_quick.id.value,
    F.Train_Oracle_quick.id.value,
    # F.Train_Tempest_quick.id.value,
    F.Train_WarpPrism_quick.id.value,
    F.Train_Colossus_quick.id.value,
    F.Train_Immortal_quick.id.value,
    F.Train_Disruptor_quick.id.value,
    F.TrainWarp_Zealot_pt.id.value,
    F.TrainWarp_Stalker_pt.id.value,
    F.TrainWarp_HighTemplar_pt.id.value,
    F.TrainWarp_DarkTemplar_pt.id.value,
    # F.TrainWarp_Sentry_pt.id.value,
    # F.TrainWarp_Adept_pt.id.value,
    F.Morph_Archon_quick.id.value,
    F.Morph_WarpGate_quick.id.value,
    F.Research_WarpGate_quick.id.value,
]

LARGE_LIST = [   
    F.Train_Probe_quick.id.value,
    F.Build_Nexus_pt.id.value,
    F.Build_Pylon_pt.id.value,
    F.Build_Gateway_pt.id.value,
    F.Build_Assimilator_unit.id.value,
    F.Build_CyberneticsCore_pt.id.value,
    F.Build_Forge_pt.id.value,
    F.Build_FleetBeacon_pt.id.value,
    F.Build_TwilightCouncil_pt.id.value,
    F.Build_PhotonCannon_pt.id.value,
    F.Build_Stargate_pt.id.value,
    F.Build_TemplarArchive_pt.id.value,
    F.Build_DarkShrine_pt.id.value,
    F.Build_RoboticsBay_pt.id.value,
    F.Build_RoboticsFacility_pt.id.value,
    F.Build_ShieldBattery_pt.id.value,
    F.Train_Zealot_quick.id.value,
    F.Train_Stalker_quick.id.value,
    F.Train_HighTemplar_quick.id.value,
    F.Train_DarkTemplar_quick.id.value,
    F.Train_Sentry_quick.id.value,
    F.Train_Adept_quick.id.value,
    F.Train_Phoenix_quick.id.value,
    F.Train_Carrier_quick.id.value,
    F.Train_VoidRay_quick.id.value,
    F.Train_Oracle_quick.id.value,
    F.Train_Tempest_quick.id.value,
    F.Train_WarpPrism_quick.id.value,
    F.Train_Colossus_quick.id.value,
    F.Train_Immortal_quick.id.value,
    F.Train_Disruptor_quick.id.value,
    F.TrainWarp_Zealot_pt.id.value,
    F.TrainWarp_Stalker_pt.id.value,
    F.TrainWarp_HighTemplar_pt.id.value,
    F.TrainWarp_DarkTemplar_pt.id.value,
    F.TrainWarp_Sentry_pt.id.value,
    F.TrainWarp_Adept_pt.id.value,
    F.Morph_Archon_quick.id.value,
    F.Morph_WarpGate_quick.id.value,
    F.Research_WarpGate_quick.id.value,
    F.Research_Charge_quick.id.value,
    F.Research_Blink_quick.id.value,
    F.Research_AdeptResonatingGlaives_quick.id.value,
    F.Research_ShadowStrike_quick.id.value,
    F.Research_ProtossAirArmor_quick.id.value,
    F.Research_ProtossAirWeapons_quick.id.value,
    F.Research_ProtossGroundArmor_quick.id.value,
    F.Research_ProtossGroundWeapons_quick.id.value,
    F.Research_ProtossShields_quick.id.value,
    F.Research_GraviticBooster_quick.id.value,
    F.Research_GraviticDrive_quick.id.value,
    F.Research_ExtendedThermalLance_quick.id.value,
    F.Research_PsiStorm_quick.id.value,
    F.Effect_ChronoBoostEnergyCost_unit.id.value,
    F.Effect_ChronoBoost_unit.id.value,
    F.Effect_GravitonBeam_unit.id.value,
    F.Effect_PsiStorm_pt.id.value,
    F.Effect_ForceField_pt.id.value,
    F.Effect_OracleRevelation_pt.id.value,
    F.Effect_AdeptPhaseShift_pt.id.value,
    F.Effect_Feedback_unit.id.value,
    F.Effect_Restore_unit.id.value,
    F.Effect_PurificationNova_pt.id.value,
    F.Effect_Blink_pt.id.value,
    F.Effect_Blink_unit.id.value,
    F.Effect_MassRecall_pt.id.value,
    F.Effect_VoidRayPrismaticAlignment_quick.id.value,
    F.Effect_Repair_pt.id.value,
    F.Effect_Repair_unit.id.value,
    F.Behavior_PulsarBeamOn_quick.id.value,
    F.Behavior_PulsarBeamOff_quick.id.value,
]


def small_select_and_target_unit_type_for_actions(func_name):

    select = None
    target = None
    min_num = None

    try:
        if func_name == "Train_Probe_quick":
            select = Protoss.Nexus

        elif func_name == "Build_Pylon_pt":
            select = Protoss.Probe

        elif func_name == "Build_Gateway_pt":
            select = Protoss.Probe

        elif func_name == "Build_Assimilator_unit":
            select = Protoss.Probe
            target = Neutral.VespeneGeyser

        elif func_name == "Build_CyberneticsCore_pt":
            select = Protoss.Probe

        elif func_name == "Train_Zealot_quick":
            select = Protoss.Gateway

        elif func_name == "Train_Stalker_quick":
            select = Protoss.Gateway

    except Exception as e:
        print("Find exception in small_select_and_target_unit_type_for_actions:", e)
        print(traceback.format_exc())

    finally:
        return select, target, min_num


def select_and_target_unit_type_for_protoss_actions(function_call):

    select = None
    target = None
    min_num = None

    try:
        function = function_call.function

        # print('function', function)
        # print('F[function]', F[function])
        # print('F.Train_Probe_quick', F.Train_Probe_quick)

        if F[function] == F.Train_Probe_quick:
            select = Protoss.Nexus

        elif F[function] == F.Build_Pylon_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_Gateway_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_Assimilator_unit:
            select = Protoss.Probe
            target = Neutral.VespeneGeyser

        elif F[function] == F.Build_CyberneticsCore_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_Forge_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_PhotonCannon_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_ShieldBattery_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_TwilightCouncil_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_TemplarArchive_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_DarkShrine_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_Stargate_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_FleetBeacon_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_RoboticsFacility_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_RoboticsBay_pt:
            select = Protoss.Probe

        elif F[function] == F.Build_Nexus_pt:
            select = Protoss.Probe

        elif F[function] == F.Train_Zealot_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_Zealot_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Train_Stalker_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_Stalker_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Train_Sentry_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_Sentry_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Train_Adept_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_Adept_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Train_HighTemplar_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_HighTemplar_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Train_DarkTemplar_quick:
            select = Protoss.Gateway

        elif F[function] == F.TrainWarp_DarkTemplar_pt:
            select = Protoss.WarpGate

        elif F[function] == F.Morph_Archon_quick:
            select = [Protoss.HighTemplar, Protoss.DarkTemplar]
            min_num = 2  # we must need 2 units to morph a Archon

        elif F[function] == F.Train_Phoenix_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_Oracle_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_VoidRay_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_Tempest_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_Carrier_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_Mothership_quick:
            select = Protoss.Stargate

        elif F[function] == F.Train_MothershipCore_quick:  # note, is was removed in patch 4.0.0
            select = Protoss.Nexus

        elif F[function] == F.Train_Observer_quick:
            select = Protoss.RoboticsFacility

        elif F[function] == F.Train_WarpPrism_quick:
            select = Protoss.RoboticsFacility

        elif F[function] == F.Train_Immortal_quick:
            select = Protoss.RoboticsFacility

        elif F[function] == F.Train_Colossus_quick:
            select = Protoss.RoboticsFacility

        elif F[function] == F.Train_Disruptor_quick:
            select = Protoss.RoboticsFacility

        elif F[function] == F.Morph_WarpPrismPhasingMode_quick:
            select = Protoss.WarpPrism

        elif F[function] == F.Morph_WarpPrismTransportMode_quick:
            select = Protoss.WarpPrismPhasing

        elif F[function] == F.Build_Interceptors_quick:
            select = Protoss.Carrier

        elif F[function] == F.Build_Interceptors_autocast:
            select = Protoss.Carrier

        elif F[function] == F.Rally_Nexus_pt:
            select = Protoss.Nexus

        elif F[function] == F.Rally_Nexus_unit:
            select = Protoss.Nexus

        elif F[function] == F.Effect_ChronoBoostEnergyCost_unit:  # new 4.0?
            select = Protoss.Nexus

        elif F[function] == F.Effect_ChronoBoost_unit:  # wrong / old?
            select = Protoss.Nexus

        elif F[function] == F.Effect_Blink_pt:
            select = Protoss.Stalker

        elif F[function] == F.Effect_Blink_Stalker_pt:
            select = Protoss.Stalker

        elif F[function] == F.Effect_ForceField_pt:
            select = Protoss.Sentry

        elif F[function] == F.Effect_AdeptPhaseShift_pt:
            select = Protoss.Adept

        elif F[function] == F.Effect_PsiStorm_pt:
            select = Protoss.HighTemplar

        elif F[function] == F.Research_PsiStorm_quick:
            select = Protoss.TemplarArchive

        elif F[function] == F.Effect_ShadowStride_pt:  # new in 3.8.0
            select = Protoss.DarkTemplar

        elif F[function] == F.Research_ShadowStrike_quick:  # interesting, it was spelled wrongly in pysc2-3.0. "strike" should be "stride".
            select = Protoss.DarkShrine

        elif F[function] == F.Effect_GravitonBeam_unit:
            select = Protoss.Phoenix

        elif F[function] == F.Cancel_GravitonBeam_quick:
            select = Protoss.Phoenix

        elif F[function] == F.Behavior_PulsarBeamOn_quick:
            select = Protoss.Oracle

        elif F[function] == F.Behavior_PulsarBeamOff_quick:
            select = Protoss.Oracle

        elif F[function] == F.Build_Interceptors_autocast:
            select = Protoss.Carrier

        elif F[function] == F.Build_Interceptors_quick:
            select = Protoss.Carrier

        elif F[function] == F.Load_unit:
            select = Protoss.WarpPrism

        elif F[function] == F.Load_WarpPrism_unit:
            select = Protoss.WarpPrism

        elif F[function] == F.UnloadAllAt_WarpPrism_pt:
            select = Protoss.WarpPrism

        elif F[function] == F.UnloadAllAt_WarpPrism_unit:
            select = Protoss.WarpPrism

        elif F[function] == F.UnloadUnit_WarpPrism_quick:
            select = Protoss.WarpPrism

        elif F[function] == F.Effect_PurificationNova_pt:
            select = Protoss.Disruptor

        elif F[function] == F.Effect_MassRecall_Mothership_pt:
            select = Protoss.Mothership

        elif F[function] == F.Effect_MassRecall_Nexus_pt:
            select = Protoss.Nexus

        elif F[function] == F.Effect_MassRecall_pt:
            select = [Protoss.Nexus, Protoss.Mothership]           

        elif F[function] == F.Effect_MassRecall_StrategicRecall_pt:
            select = [Protoss.Nexus, Protoss.Mothership] 

        elif F[function] == F.Research_WarpGate_quick:
            select = Protoss.CyberneticsCore

        elif F[function] == F.Research_Charge_quick:
            select = Protoss.TwilightCouncil

        elif F[function] == F.Research_Blink_quick:
            select = Protoss.TwilightCouncil

        elif F[function] == F.Research_AdeptResonatingGlaives_quick:
            select = Protoss.TwilightCouncil

        elif F[function] == F.Research_ProtossGroundWeaponsLevel1_quick:
            select = Protoss.Forge

    except Exception as e:
        print("Find exception in select_and_target_unit_type_for_protoss_actions:", e)
        print(traceback.format_exc())

    finally:
        return select, target, min_num
