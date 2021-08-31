Notice:
-------
If you are making a Mini-app from an existing software project, do not name the mini-app after the parent software package.
This can be confusing to everyone involved in the release chain that you are trying to release an open source version of X.


Steps so far
- Get an SACR
  - Respectfully email leslie.j.johnson@nasa.gov and ask about steps for open source software release
  - She will email you back a thorough description of Safety classifications and a list of questions
  - Email her back the answers to her questions based on the project you are looking to release.
  - She will send you back an initial SACR.
  - You now need an RMM to send back to her. 

- Get an RMM
  - Once you have an initial SACR from Leslie you will have a classification (D, or E).
  - Get the appropriate example from:  https://gitlab.larc.nasa.gov/mpark/casb-sw-mgmt-plan
  - Rename the file to be `RMM_<project>.xlsx`
  - Read through the document, but likely you will just need to alter the CASB required column j. This is the column of "evidence" that you are, in fact, taking action that makes the project fully compliant.
  - "Sign" this excel file by typing in your name and date.
  - Send this to Beth to sign.
  - Then email the signed copy to Leslie.
  - She will email you back the RMM after she signs it.

- NTR
  - Go to https://invention.nasa.gov/ to create an initial NTR
  - Fill it out
  - Submit it
  - Tell any co-authors they need to "review it" by logging in and clicking through.  For any required comment box they can enter "correct to my knowledge".
  - Log back into NTR and "submit" it.

- Patent Rights Questionaire (NF343)
  - Wait 12ish hours after you submit your NTR
  - Add your LAR-XXXXX-X number in the top right
  - Put in the project name (copy from NTR)
  - Get all inventors to sign under the "Agreement to Assign".  This yields all patent rights for the mini-app to the government.  Since your mini-app should be based on openly published techniques, nothing should be patentable anyway.
  - Have all inventors email this back to Elaine.C.McMahon@nasa.gov

- Software Release System (SRS)
  - To move onto the SRS you should have your signed RMM back from Leslie, and have submitted to the NTR.
  - Bonnie will likely email you a nudge to start your submission on the SRS.  If not, you can go to https://softwarerelease.ndc.nasa.gov/ to do it. 
  - You need 3 steps to continue.  Your signed RMM. Answer a few questions about security.  508 compliance.
  - Your Mini-app likely has no human interface, so you are 508 compliant.
  - For the security questions, answer them honestly, though since the mini-app shouldn't do much, you will be "self consulting" that it doesn't monitor network traffic, contain PII, etc.
  - Then upload the signed RMM in that section.
  - Then submit the SRS.
  - From there it gets sent to Beth to sign, before going through all the steps posted below (export control, commercialization, on and on)
