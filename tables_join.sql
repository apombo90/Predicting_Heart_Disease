SELECT
	binary_data.id,
	binary_data.heartdisease,
	binary_data.smoking,
	binary_data.alcoholdrinking,
	binary_data.stroke,
	binary_data.diffwalking,
	binary_data.sex,
	binary_data.physicalactivity,
	binary_data.asthma,
	binary_data.kidneydisease,
	binary_data.skincancer,
	nonbinary_data.bmi,
	nonbinary_data.physicalhealth,
	nonbinary_data.mentalhealth,
	nonbinary_data.agecategory,
	nonbinary_data.race,
	nonbinary_data.diabetic,
	nonbinary_data.genhealth,
	nonbinary_data.sleeptime	
FROM binary_data
	INNER JOIN nonbinary_data
	ON binary_data.id = nonbinary_data.id
ORDER BY
	binary_data.id;
