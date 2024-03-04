# Generative Designa and Manufacturing
**Workflow**
![image](https://github.com/AdityaJoglekar/Generative_Design_and_Manufacturing/assets/92458082/bb3039e3-ff6f-4e21-b97b-b7802955fb08)
Our proposed framework. 1) The user inputs the problem domain and boundary conditions and a set of manufacturing method and material combinations. 2) An initial probe of the supply chain using system generated representative part guesses helps swift removal of infeasible combinations. This process also helps establish the relationship between different requirements (mass, compliance, lead time, cost) for each of the current suppliers, gives an approximate range of values for each of the requirements that help the user determine the constraints for mass, lead time and cost for performing topology optimization in the design generator and also helps determine the best supplier to achieve Pareto optimal solutions. 3) One optimized design corresponding to each input configuration (defined by boundary conditions, manufacturing
method, material and supplier) is output by the design generator. 4) These designs are then passed through the supply chain scheduler to get the lead time and cost and explore the trade-offs for different suppliers. 5) All the results are then evaluated and visualized using the Explainable AI and Results Interface, which helps in user feedback to the system and selection of the most effective part.

**Design Generator**
![image](https://github.com/AdityaJoglekar/Generative_Design_and_Manufacturing/assets/92458082/6b4a2e5f-ed33-449a-885a-9fff659f2e1f)

