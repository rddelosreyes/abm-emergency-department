import mesa

class PatientAgent(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Patient characteristics
        self.acuity = ''
        self.disposition = ''
        self.complexity = ''
        self.destination_record = ''  # trajectory

        # Length of stay
        self.ed_los = 0
        self.execution_los = 0
        self.waiting_los = 0

        # For simulation purposes
        self.destination_list = []
        self.execution_list = []
        self.execution_ctr = 0
        self.in_execution_queue = 0

        # For recording purposes
        self.arrival_time_in_days = -1

    def step(self):
        if self.in_execution_queue:
            self.ed_los += 1
            self.execution_ctr -= 1
            self.execution_los += 1
        else:
            self.ed_los += 1
            self.waiting_los += 1