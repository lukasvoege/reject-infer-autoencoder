# reject-infer-autoencoder
Information Systems Seminar Group 1

# Notes
- MMD Loss braucht große batch sizes, MMSE leider eher kleine
- MMD bringt 0, verschlechtert situation nur
- KLVDiv funktioniert, sorgt für weniger sampling bias

### Ideas why weighting has no effect at all:
- Balancing the autoencoder train set does the main trick already
- KLDiv loss diverges after 1 itteration, making all subsequent weighting useless

# Ideas 
- streamlit
--> options for different metrics
- Categorical Features: WOE or OneHot before Autoencoder?



# TO-DOs
! --> Loss Function muss unbedingt irgendwie Accpets vs Rejects vergleichen, sonst machts kein Sinn alles

- WOE encode alles categorical
- Fix DecisonTree simulation Loop
- Andere Methoden scouten / vllt. implementieren / testen
- Simulation mal mit Encodeten Daten probieren
- Folien bauen


user: bernd
Pw:BerndStromberg1!