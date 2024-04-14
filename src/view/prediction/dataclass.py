from dataclasses import dataclass


@dataclass
class Prediction:
    id: None = 1
    customer_id: str = ''
    label: int = 1
    percentage: int = 1

    def as_dict(self):
        return {
            'customer_id': self.customer_id,
            'label': self.label,
            'percentage': self.percentage
        }
