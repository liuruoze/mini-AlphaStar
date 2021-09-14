# mini-AlphaStar


## v_1.03

* Add guides for "how run RL?" in USAGE.MD;
* Fix some bugs and a bug in SL training;
* Add USAGE.MD;
* Fix a bug by split the map_channels and scatter_channels;
* Fix the right version to see replays of AlphaStar Final Terran;
* Change from z_2 = F.relu(self.fc_2(z_1)) to z_2 = self.fc_2(F.relu(z_1)) in selected_units_head;

## v_1.02

* Add relu after each downsampling conv2d in spatial encoder;
* Change the bias of most conv and convtranspose from False to True (if not has a bn after or in 3rd lib);
* Add "masked by the missing entries" in entity_encoder;
* Formalized a lib function to be used: unit_tpye_to_unit_type_index;
* Fix a TODO in calculate_unit_counts_bow() by unit_tpye_to_unit_type_index();
* Change back to use All_Units_Size equals to all unit_type_index;
* Add scatter entities map in spatial_encoder;

## v_1.01

* Change to using the win-loss reward in winloss_baseline (the thing AlphaStar should do);
* Refine pseudo reward, scale Leven reward, right log prob ( which should be negative CrossEntropy);
* Fix some problems due to wrong original codes of AlphaStar;
* Add analysis of move camera count in the AlphaStar replay;

## v_1.00

* Make some improvements for SL training；
* Fix some bug in SL training；
* Fix a RL training bug due to the cudnn of GPU and regroup the directory;
* Add time decay scale for unit count reward (hamming distance);

## v_0.99

* Add the code for analyzing the statistics of replays;
* Implement the RL training with the z reward (from the experts' statistics in replays);
* Replenish the entropy_loss_for_all_arguments;
* True implementation of teacher_logits in human_policy_kl_loss and test all the above RL training on server;
* Decouple the sl_loss with the agent, and pass the test of SL training on the server;
* Fix some warnings due to PyTorch 1.5, and fill up many TODOs;

## v_0.96

* Fix the clipped_rhos calculation problem for v-trace policy gradient loss;
* Fix the filter_by function for the other 5 arguments in the vtrace_pg_loss;
* Implement the vtrace_advantages method for the all 6 arguments of actions;
* Complete the implementation of split_vtrace_pg_loss for all arguments;
* Replenish the right process flow for baselinse of build_order, built_units, upgrades, effects;
* Fix the 5 other arguments loss calculation in the UPGO;

## v_0.93

* Add the implement the unit_counts_bow;
* Add the implementation of the build_order;
* Implement upgrade, effect, last and available action, plus time coding;
* Add reward calculation for build order and unit counts
* Add the calculation of td_lambda_loss based on the reward of the Levenstein and Hamming;

## v_0.9

* Fix the eval problem in SL, add support that to use SL model to fine-tune RL training;
* Fix the too slow speed problem of the SL training;
* Fix the Rl training problem and prepared the code of testing against the built-in AI;

## v_0.8

* Add code for Levenshtein and Hamming distance;
* Add the function for fighting against built-in AI computer;
* Fix the SL training loss problem for select units;

## v_0.7

* We release the mini-AlphaStar project (v_0.7), which is a mini source version of the original AlphaStar program by DeepMind. 
* "v_0.7" means we think we have implemented above 70 percent code of it. 
* "mini" means that we make the original AlphaStar hyperparameter adjustable so that it can run on a small scale.