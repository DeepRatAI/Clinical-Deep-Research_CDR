"""
CDR RoB2 Layer

Risk of Bias 2 assessment (RoB2 for RCTs, ROBINS-I for observational).
"""

from cdr.rob2.assessor import RoB2Assessor
from cdr.rob2.robins_i_assessor import ROBINSIAssessor

__all__ = ["RoB2Assessor", "ROBINSIAssessor"]
