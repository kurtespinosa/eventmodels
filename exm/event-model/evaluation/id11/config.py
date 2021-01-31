# configuration for the BioNLP ST 2011 ID task.

entity_types = set(["Protein", "Chemical", "Organism", "Two-component-system", "Regulon-operon", "Entity"])

given_types = set(["Protein", "Chemical", "Organism", "Two-component-system", "Regulon-operon"])

event_types  = set(["Binding",
                    "Gene_expression", "Transcription", "Protein_catabolism",
                    "Regulation", "Positive_regulation", "Negative_regulation",
                    "Phosphorylation",
                    "Localization",
                    "Process"])

output_event_type_order = ['Gene_expression',
                           'Transcription',
                           'Protein_catabolism',
                           'Phosphorylation',
                           'Localization',
                           ' =[SIMPLE-TOTAL]= ',
                           'Binding',
                           'Process',
                           '==[NONREG-TOTAL]==',
                           'Regulation',
                           'Positive_regulation',
                           'Negative_regulation',
                           ' ==[REG-TOTAL]==  ',
                           ' ====[TOTAL]====  ',
                           ]

subtotal_event_set = {
    ' =[SIMPLE-TOTAL]= ': ['Gene_expression',
                           'Transcription',
                           'Protein_catabolism',
                           'Phosphorylation',
                           'Localization'],
    '==[NONREG-TOTAL]==' : ['Gene_expression',
                            'Transcription',
                            'Protein_catabolism',
                            'Phosphorylation',
                            'Localization',
                            'Binding',
                            'Process'
                            ],
    ' ==[REG-TOTAL]==  ' : ['Regulation',
                            'Positive_regulation',
                            'Negative_regulation',
                            ],
    ' ====[TOTAL]====  ' : ['Gene_expression',
                            'Transcription',
                            'Protein_catabolism',
                            'Phosphorylation',
                            'Localization',
                            'Binding',
                            'Process',
                            'Regulation',
                            'Positive_regulation',
                            'Negative_regulation',
                            ]    
}
