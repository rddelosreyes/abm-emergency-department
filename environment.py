import mesa
import networkx as nx
import numpy as np
import pandas as pd
import random
import string

from agent import PatientAgent

class HospitalResource():

    def __init__(self, identifier, name, capacity=float('inf')):
        self.identifier = identifier
        self.name = name
        self.execution_queue = []
        self.waiting_queue = []
        self.capacity = capacity

    def transfer(self):
        patient_execute = self.waiting_queue.pop(0)
        self.execution_queue.append(patient_execute)
        patient_execute.in_execution_queue = 1

class EDModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, **params):
        # Simulation run parameters
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        self.simulation_type = params['simulation_type']
        self.max_steps = params['max_steps']

        self.increase_retrospective = params['increase_retrospective']
        self.increase_type = params['increase_type']

        # ED resource parameters
        self.no_beds = params['no_beds']
        self.no_clinicians = params['no_clinicians']
        self.no_imaging = params['no_imaging']

        self.available_clinicians = params['no_clinicians']
        self.busy_clinicians = 0

        self.new_patient_id = 1
        self.hosp_waiting_1 = []
        self.hosp_waiting_2 = []
        self.hosp_waiting_345 = []
        self.hosp_execution_queue = []

        # Arrival parameters
        self.hourly_arrival_rate = params['hourly_arrival_rate']
        self.new_patient_arrival_time = np.random.poisson(60/self.hourly_arrival_rate[24])

        # Process parameters
        self.branching_type = params['branching_type']  # independent or conditional
        self.cohort_type = params['cohort_type']  # prospective or retrospective
        process_model_dict = params['process_model']
        path_process_time = params['path_process_time']
        path_process_branching = params['path_process_branching']
        path_cohort_frequency = params['path_cohort_frequency']

        assert self.cohort_type in ['prospective', 'retrospective'], 'Select cohort type from the following options: prospective, retrospective'
        self.categories_acuity = ['1', '2', '3', '4', '5']

        process_model = self._get_process_model_graph(process_model_dict)

        self.process_dict = {}
        self.process_shared = {}
        self.process_unshared = {}
        for process_id in process_model.nodes():
            new_process = HospitalResource(process_id, process_model.nodes[process_id]['name'], process_model.nodes[process_id]['capacity'])
            self.process_dict[process_id] = new_process

            if process_model.nodes[process_id]['share_clinicians']:
                self.process_shared[process_id] = new_process
            else:
                self.process_unshared[process_id] = new_process

        self.dict_branching_prob_list = {}
        self.dict_execution_time_list = {}

        self.df_frequency_subcohort = pd.read_csv(f'{path_cohort_frequency}/frequency_groupname.csv', index_col='acuity').sort_values(['acuity']).reset_index()
        self.df_frequency_subcohort['groupname'] = self.df_frequency_subcohort['acuity'].astype(str) + '-' + self.df_frequency_subcohort['disposition'].astype(str) + '-' + self.df_frequency_subcohort['complexity'].astype(str)
        groupname_list = set(self.df_frequency_subcohort['groupname'])

        if self.cohort_type == 'prospective':
            for acuity_val in self.categories_acuity:
                branch_prob = pd.read_csv(f'{path_process_branching}/{self.branching_type}_branching_acuity_{acuity_val}.csv', index_col='name').to_dict(orient='index')
                fit_time = pd.read_csv(f'{path_process_time}/time_acuity_{acuity_val}.csv', index_col='edge').to_dict(orient='index')

                self.dict_branching_prob_list[acuity_val] = branch_prob
                self.dict_execution_time_list[acuity_val] = fit_time

        elif self.cohort_type == 'retrospective':
            for groupname_val in groupname_list:
                branch_prob = pd.read_csv(f'{path_process_branching}/{self.branching_type}_branching_groupname_{groupname_val}.csv', index_col='name').to_dict(orient='index')
                fit_time = pd.read_csv(f'{path_process_time}/time_groupname_{groupname_val}.csv', index_col='edge').to_dict(orient='index')

                self.dict_branching_prob_list[groupname_val] = branch_prob
                self.dict_execution_time_list[groupname_val] = fit_time

        self.dict_frequency_per_acuity = {}
        for acuity_val, disp_val, aecc_val, per_val in zip(self.df_frequency_subcohort['acuity'], self.df_frequency_subcohort['disposition'], self.df_frequency_subcohort['complexity'], self.df_frequency_subcohort['percent']):
            acuity_val = str(acuity_val)
            if acuity_val not in self.dict_frequency_per_acuity.keys():
                self.dict_frequency_per_acuity[acuity_val] = {}

            grp_val = disp_val + '-' + aecc_val
            self.dict_frequency_per_acuity[acuity_val][grp_val] = per_val

        self.df_frequency_acuity = pd.read_csv(f'{path_cohort_frequency}/frequency.csv')

        if self.increase_retrospective != 0:
            self.df_frequency_subcohort_increased = pd.read_csv(f'{path_cohort_frequency}/frequency_groupname_{self.increase_type}_{self.increase_retrospective}.csv', index_col='acuity').sort_values(['acuity']).reset_index()

            self.df_frequency_subcohort_increased['groupname'] = self.df_frequency_subcohort_increased['acuity'].astype(str) + '-' + self.df_frequency_subcohort_increased['disposition'].astype(str) + '-' + self.df_frequency_subcohort_increased['complexity'].astype(str)
            groupname_list = set(self.df_frequency_subcohort_increased['groupname'])

            self.dict_frequency_per_acuity_increased = {}
            for acuity_val, disp_val, aecc_val, per_val in zip(self.df_frequency_subcohort_increased['acuity'], self.df_frequency_subcohort_increased['disposition'], self.df_frequency_subcohort_increased['complexity'], self.df_frequency_subcohort_increased['percent']):
                acuity_val = str(acuity_val)
                if acuity_val not in self.dict_frequency_per_acuity_increased.keys():
                    self.dict_frequency_per_acuity_increased[acuity_val] = {}

                grp_val = disp_val + '-' + aecc_val
                self.dict_frequency_per_acuity_increased[acuity_val][grp_val] = per_val

        # Mesa parameters
        self.grid = mesa.space.NetworkGrid(process_model)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Output parameters
        self.runin_period = params['runin_period']
        self.output_folder = params['output_folder']

        model_reporters = self._set_model_outputs_to_collect()
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
        )

        self.synthetic_ehrs = {}

    def step(self):
        """Advance the model by one step."""
        # Add new patients based on hourly arrival rate
        while self.new_patient_arrival_time == 0:
            self._create_new_arrival()
            hour_of_day = self._get_hour_of_day()

            if self.increase_type != 'none':
                day_now = self._get_time_in_days()

                magnitude_increase_val = 0
                if 7 <= day_now < 11:
                    magnitude_increase_val = 1

                self.new_patient_arrival_time = np.random.poisson(60/(self.hourly_arrival_rate[hour_of_day] + magnitude_increase_val))
            else:
                self.new_patient_arrival_time = np.random.poisson(60/self.hourly_arrival_rate[hour_of_day])

        # Move patients from hospital waiting queue to hospital execution queue if bed is available
        if self._get_bed_occupancy() < self.no_beds:
            number_of_patients_to_add = self.no_beds - self._get_bed_occupancy()
            in_waiting_queue = self.hosp_waiting_1 + self.hosp_waiting_2 + self.hosp_waiting_345
            patients_to_add = in_waiting_queue[:number_of_patients_to_add]

            for waiting_patient in patients_to_add:
                if waiting_patient.acuity == 1:
                    self.hosp_waiting_1.remove(waiting_patient)
                elif waiting_patient.acuity == 2:
                    self.hosp_waiting_2.remove(waiting_patient)
                else:
                    self.hosp_waiting_345.remove(waiting_patient)

                self.hosp_execution_queue.append(waiting_patient)

                next_resource = waiting_patient.destination_list.pop(0)
                self.grid.place_agent(waiting_patient, next_resource)
                self.process_dict[next_resource].execution_queue.append(waiting_patient)
                waiting_patient.in_execution_queue = 1

                next_execution = waiting_patient.execution_list.pop(0)
                waiting_patient.execution_ctr = next_execution

        # Move all patients
        for patient in self.schedule.agents:
            patient.step()

        # Transfer patients in bed across hospital resources
        for resource_unit in list(self.process_dict.values())[::-1]:
            for patient_in_bed in list(resource_unit.execution_queue):
                if patient_in_bed.execution_ctr == 0:
                    resource_unit.execution_queue.remove(patient_in_bed)

                    if len(patient_in_bed.destination_list) > 1:
                        next_resource = patient_in_bed.destination_list.pop(0)
                        self.grid.move_agent(patient_in_bed, next_resource)
                        self.process_dict[next_resource].waiting_queue.append(patient_in_bed)
                        patient_in_bed.in_execution_queue = 0

                        next_execution = patient_in_bed.execution_list.pop(0)
                        patient_in_bed.execution_ctr = next_execution
                    else:
                        if self.increase_type != 'none':
                            if self.runin_period <= patient_in_bed.arrival_time_in_days < 11:
                                self._record_removed_patient(patient_in_bed)
                        else:
                            if patient_in_bed.arrival_time_in_days >= self.runin_period:
                                self._record_removed_patient(patient_in_bed)

                        self.hosp_execution_queue.remove(patient_in_bed)
                        self.grid.remove_agent(patient_in_bed)
                        self.schedule.remove(patient_in_bed)

                    if resource_unit.identifier in self.process_shared.keys():
                        self.available_clinicians += 1
                        self.busy_clinicians -= 1

        # Move patients in bed from waiting queue to execution queue per resource unit if space becomes available
        for resource_unit in self.process_unshared.values():
            while len(resource_unit.execution_queue) < resource_unit.capacity:
                if resource_unit.waiting_queue:
                    resource_unit.transfer()
                else:
                    break

        while self.available_clinicians:
            shared_resources_with_in_waiting_queue = [resource_unit for resource_unit in self.process_shared.values() if (len(resource_unit.waiting_queue) > 0) and (len(resource_unit.execution_queue) < resource_unit.capacity)]

            if shared_resources_with_in_waiting_queue:
                random.shuffle(shared_resources_with_in_waiting_queue)

                shared_resources_with_in_waiting_queue[0].transfer()
                self.available_clinicians -= 1
                self.busy_clinicians += 1
            else:
                break

        assert self.busy_clinicians <= self.no_clinicians, 'Number of occupied clinicians should be at most equal to the number of clinicians'

        self.schedule.steps += 1
        self.schedule.time += 1
        self.new_patient_arrival_time -= 1

        if self.schedule.time == self.max_steps:
            self.synthetic_ehrs = pd.DataFrame.from_dict(self.synthetic_ehrs, orient='index')
            self.synthetic_ehrs.to_csv(f'{self.output_folder}/seed_{self._seed}.csv', index=False)

            self.datacollector.collect(self)
            self.running = False

    def _set_model_outputs_to_collect(self):
        self.time_los_list = []
        self.time_execution_list = []
        self.time_waiting_list = []

        model_reporters = {'LOS': 'time_los_list',
                           'Execution time': 'time_execution_list',
                           'Waiting time': 'time_waiting_list',
                           }

        return model_reporters

    def _get_day_of_week(self):
        day_of_week = int(self._get_time_in_days()) % 7
        if day_of_week == 0: day_of_week = 7

        return day_of_week

    def _get_hour_of_day(self):
        hour_of_day = int(self._get_time_in_hours()) % 24
        if hour_of_day == 0: hour_of_day = 24

        return hour_of_day

    def _get_bed_occupancy(self):

        return len(self.hosp_execution_queue)

    def _get_bed_waiting(self):

        return len(self.hosp_waiting_1) + len(self.hosp_waiting_2) + len(self.hosp_waiting_345)

    def _get_time_in_days(self):

        return self.schedule.time / (60*24)

    def _get_time_in_hours(self):

        return self.schedule.time / 60

    def _get_process_model_graph(self, process_model_dict):
        # Assign alphabet IDs to process names (for better handling)
        node_dict = {}
        alphabet_counterpart = string.ascii_uppercase[:len(process_model_dict)]
        for process_id, (process_name, process_attributes) in zip(alphabet_counterpart, process_model_dict.items()):
            node_dict[process_id] = {'name': process_name, 'capacity': float(process_attributes['capacity']), 'share_clinicians': process_attributes['share_clinicians']}

        # Create process model
        process_model = nx.complete_graph(node_dict.keys(), nx.DiGraph())
        nx.set_node_attributes(process_model, node_dict)

        return process_model

    def _create_new_arrival(self):
        new_patient = PatientAgent(self.new_patient_id, self)

        if self.increase_type != 'none':
            day_now = self._get_time_in_days()

            frequency_increase_val = 0
            if 7 <= day_now < 11:
                frequency_increase_val = 0.10

            acuity_name_list = self.df_frequency_acuity.acuity.tolist()
            baseline_frequency_list = self.df_frequency_acuity.percent.tolist()
            new_frequency_list = []

            high_acuity_frequency = sum(baseline_frequency_list[:2])
            low_acuity_frequency = sum(baseline_frequency_list[2:])

            for acuity_idx, _ in enumerate(acuity_name_list):
                if acuity_idx in [0, 1]:
                    new_frequency_val = (high_acuity_frequency + frequency_increase_val) * (baseline_frequency_list[acuity_idx] / high_acuity_frequency)
                else:
                    new_frequency_val = (low_acuity_frequency - frequency_increase_val) * (baseline_frequency_list[acuity_idx] / low_acuity_frequency)
                new_frequency_list.append(new_frequency_val)

            new_patient.acuity = str(random.choices(self.df_frequency_acuity.acuity.tolist(), weights=new_frequency_list)[0])
        else:
            new_patient.acuity = str(random.choices(self.df_frequency_acuity.acuity.tolist(), weights=self.df_frequency_acuity.percent.tolist())[0])
            patient_end_weights = list(self.dict_frequency_per_acuity[new_patient.acuity].values())

        if self.increase_retrospective != 0:
            day_now = self._get_time_in_days()

            if 7 <= day_now < 11:
                patient_end_weights = list(self.dict_frequency_per_acuity_increased[new_patient.acuity].values())
            else:
                patient_end_weights = list(self.dict_frequency_per_acuity[new_patient.acuity].values())
        else:
            patient_end_weights = list(self.dict_frequency_per_acuity[new_patient.acuity].values())

        patient_end = random.choices(list(self.dict_frequency_per_acuity[new_patient.acuity].keys()), weights=patient_end_weights)[0]
        new_patient.disposition = patient_end.split('-')[0]
        new_patient.complexity = patient_end.split('-')[1]

        if self.cohort_type == 'prospective':
            dict_branch_prob = self.dict_branching_prob_list[new_patient.acuity]
            dict_execution_time = self.dict_execution_time_list[new_patient.acuity]
        elif self.cohort_type == 'retrospective':
            dict_branch_prob = self.dict_branching_prob_list[new_patient.acuity + '-' + new_patient.disposition + '-' + new_patient.complexity]
            dict_execution_time = self.dict_execution_time_list[new_patient.acuity + '-' + new_patient.disposition + '-' + new_patient.complexity]

        process_ids = sorted(list(self.process_dict.keys()))
        destination_list = [process_ids[0]]

        while True:
            if self.branching_type == 'independent':
                next_destination = random.choices(list(self.grid.G.nodes), weights=list(dict_branch_prob[destination_list[-1]].values()))[0]
            elif self.branching_type == 'conditional':
                ancestor_path = ''.join(destination_list)
                next_destination = random.choices(list(self.grid.G.nodes), weights=list(dict_branch_prob[ancestor_path].values()))[0]

            destination_list.append(next_destination)
            if next_destination == process_ids[-1]:
                break

        execution_list = []
        for src, dst in zip(destination_list[:-1], destination_list[1:]):
            if str(dict_execution_time[src + dst]['param']) == '0':
                exec_time = 1
            else:
                fit_param = eval(str(dict_execution_time[src + dst]['param']))
                exec_time = random.choices(list(fit_param.keys()), weights=list(fit_param.values()))[0]  # in seconds
                exec_time = max(1, round(exec_time * 60))

            execution_list.append(exec_time)

        new_patient.destination_record = ''.join(destination_list)
        new_patient.destination_list = destination_list
        new_patient.execution_list = execution_list
        new_patient.arrival_time_in_days = self._get_time_in_days()

        self.schedule.add(new_patient)
        self.new_patient_id += 1

        if self._get_bed_occupancy() == self.no_beds:
            if new_patient.acuity == 1:
                self.hosp_waiting_1.append(new_patient)
            elif new_patient.acuity == 2:
                self.hosp_waiting_2.append(new_patient)
            else:
                self.hosp_waiting_345.append(new_patient)
        else:
            self.hosp_execution_queue.append(new_patient)

            next_resource = new_patient.destination_list.pop(0)
            self.grid.place_agent(new_patient, next_resource)
            self.process_dict[next_resource].execution_queue.append(new_patient)
            new_patient.in_execution_queue = 1

            next_execution = new_patient.execution_list.pop(0)
            new_patient.execution_ctr = next_execution

    def _record_removed_patient(self, patient):
        self.time_los_list.append(patient.ed_los/60)
        self.time_execution_list.append(patient.execution_los/60)
        self.time_waiting_list.append(patient.waiting_los/60)

        patient_record = {key: val for key, val in vars(patient).items() if key not in ['model', 'pos', 'destination_list', 'execution_list', 'execution_ctr', 'in_execution_queue']}
        self.synthetic_ehrs[patient.unique_id] = patient_record