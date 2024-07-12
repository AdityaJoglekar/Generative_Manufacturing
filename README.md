# Generative Manufacturing

**Abstract**

Advances in CAD and CAM have enabled engineers and design teams to digitally design parts with unprecedented ease. Software solutions now come with a range of modules for optimizing designs for performance requirements, generating  instructions for manufacturing, and for digitally tracking the entire process from design to procurement in the form of product life-cycle management tools. However, existing solutions force design teams and corporations to take a primarily serial approach where manufacturing and procurement decisions are largely contingent on design, rather than being an integral part of the design process. In this work, we propose a new approach to part making where design, manufacturing and supply chain requirements and resources can be jointly considered and optimized for. We present the Generative Manufacturing compiler that accepts as input the following: 1) An engineering part requirements specification that includes quantities such as loads, domain envelope, mass and compliance, 2) A business part requirements specification that includes production volume, cost, and lead time, 3) Contextual knowledge about the current manufacturing state such as availability of relevant manufacturing equipment, materials and workforce, both locally and through the supply chain. Based on these factors, the compiler generates and evaluates manufacturing process alternatives and the optimal derivative designs that are implied by each process, and enables a user guided iterative exploration of the design space. As part of our initial implementation of this compiler, we demonstrate the effectiveness of our approach on examples of a cantilever beam problem and a rocket engine mounting bracket problem and showcase its utility in creating and selecting  optimal solutions according to the requirements and resources.

**Proposed Concept**
![image](https://github.com/user-attachments/assets/694cb757-5dba-4598-950b-2d18907f7afc)

**Framework**
![image](https://github.com/user-attachments/assets/79df1e9c-ae0d-43df-9dc8-e050c4100945)


**Getting Started**

Run the executable file lmco.exe present in the executable-win folder for starting the supply chain scheduler. Use run_jupyter.ipynb for running the optimization and experiments.

