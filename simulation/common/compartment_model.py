from enum import Enum
from .compartment import Compartment
class Stati(Enum):
    METABOLIZED = 1
    FREE_OR_BOUND_UNSPEC = 2
    FREE = 3
    BOUND_SPEC = 4
    BOUND_UNSPEC = 5
    BLOOD = 6

class CompartmentModel:
    def create(self, x_tissue_compartment_model, k_values, roi, blood_roi):
        self.k_values = k_values
        self.roi = roi
        self.blood_roi = blood_roi
        assert self.roi != None
        create_compartment = self._get_compartment_system(x_tissue_compartment_model)
        return create_compartment()

    def _get_compartment_system(self, n_tissues):
        if n_tissues == 1:
            assert len(self.k_values) == 2, "Not two k-values for one compartment c_1"
            return self._create_1tcm
        elif n_tissues == 2:
            assert (
                len(self.k_values) == 4 or len(self.k_values) == 3
            ), "Not four (three if k4=0) k-values for two compartments c_1,c_2"
            return self._create_2tcm
        elif n_tissues == 3:
            assert (
                len(self.k_values) == 6
            ), "Not six k-values for three compartments c_1,c_2,c_3"
            return self._create_3tcm
        else:
            raise ValueError(n_tissues)

    def _create_1tcm(self):
        c_1 = Compartment(
            k_in=self.k_values[0],
            k_outs=[self.k_values[1]],
            C_outs=[self.blood_roi],
            cell_status=Stati.METABOLIZED,
            cell_speed=0,
            roi=self.roi,
            descriptor="Compartment with concentration C1 compartment for metabolized tracer",
            type_id="C1",
        )
        return c_1

    def _create_2tcm(self):
        # simplification over 3tcm model if free <-> nonspecific is significantly faster than free <-> specific binding
        c_1 = Compartment(
            k_in=self.k_values[0],
            k_outs=[self.k_values[1]],
            C_outs=[self.blood_roi],
            cell_status=Stati.FREE_OR_BOUND_UNSPEC,
            cell_speed=0.0,
            roi=self.roi,
            descriptor="Compartment with concentration C1, free tracer and unspecificly bound tracer",
            type_id="C1",
        )
        k4 = 0
        if len(self.k_values) == 4:
            k4 = self.k_values[3]

        c_2 = Compartment(
            k_in=0,
            k_outs=[k4],
            C_outs=[c_1],
            cell_status=Stati.BOUND_SPEC,
            cell_speed=0,
            descriptor="Compartment with concentration C2 compartment for specifically bound tracer",
            type_id="C2",
        )
        c_1.add_compartment(c_2, self.k_values[2])

        return c_1

    def _create_3tcm(self):
        c_1 = Compartment(
            k_in=self.k_values[0],
            k_outs=[self.k_values[1]],
            C_outs=[self.blood_roi],
            cell_status=Stati.FREE,
            cell_speed=0.0,
            roi=self.roi,
            descriptor="Compartment with concentration C1 for tracer available in a free form for binding",
        )
        c_2 = Compartment(
            k_outs=[self.k_values[3]],
            C_outs=[c_1],
            cell_status=Stati.BOUND_UNSPEC,
            cell_speed=0,
            descriptor="Compartment with concentration C2 tracer bound to specific target molecule",
        )
        c_3 = Compartment(
            k_in=0,
            k_out=self.k_values[5],
            C_outs=[c_1],
            cell_status=Stati.BOUND_SPEC,
            cell_speed=0,
            descriptor="Compartment with concentration C3 tracer bound to unspecific molecule",
        )
        c_1.add_compartment(c_2, self.k_values[2])
        c_1.add_compartment(c_3, self.k_values[4])
        return c_1
