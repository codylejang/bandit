 Humans are barely similar to each other.                                      
  - Choice agreement: 0.509 ± 0.045 — barely above chance (0.50). Some pairs go 
  as low as 0.427.                                                              
  - NLL: 0.700 ± 0.014 — essentially at chance (0.693). One human's policy      
  barely predicts another's choices.                                            
  - Policy shape r: -0.033 ± 0.359 — centered at zero. Human psychometric curves
   don't even correlate with each other on average.                             
                                                                                
The bottom-left panel:
Individual psychometric curves go in all directions — subjects 41, 51, 55, and P43CS have negative slopes (choosing left when dQ favors right), while others have positive slopes. Some
are steep, some are flat. There's no consistent human strategy.               
                                                        
What this means for model evaluation:                                         
- The model's best composite score sits within the human-human distribution,
not below it                                                                  
- Expecting the model to match humans better than humans match each other is
not a realistic goal                                                          
- The human-human baseline is essentially chance — so any model score near    
chance isn't necessarily failing, it's hitting the ceiling set by human      
variability                                                                   
                                                        
Why humans disagree so much:                                                  
- Side biases (subject 49 picks right 63% of the time)    
- Inverted dQ sensitivity (some subjects avoid the higher-value option —      
exploration? contrarianism?)                                            
- Different exploration strategies that dQ alone doesn't capture              
                                                                
This reframes the question: instead of "does the model match the average      
human," it might be better to ask "does the model match any individual human's
strategy" — i.e., find the subject whose policy the model most resembles.