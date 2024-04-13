from dataclasses import dataclass


@dataclass
class Prediction:
    id: int
    mark: bool
    percentage: int

    def as_dict(self):
        return {
            'id': self.id,
            'mark': self.mark,
            'percentage': self.percentage
        }
