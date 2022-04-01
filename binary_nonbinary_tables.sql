-- Creating HeartDisease table
CREATE TABLE binary_data (
	id INT NOT NULL,
	HeartDisease VARCHAR NOT NULL,
	Smoking VARCHAR NOT NULL,
	AlcoholDrinking VARCHAR NOT NULL,
	Stroke VARCHAR NOT NULL,
	DiffWalking VARCHAR NOT NULL,
	Sex VARCHAR NOT NULL,
	PhysicalActivity VARCHAR NOT NULL,
	Asthma VARCHAR NOT NULL,
	KidneyDisease VARCHAR NOT NULL,
	SkinCancer VARCHAR NOT NULL,
	PRIMARY KEY (id)
);

CREATE TABLE nonbinary_data (
	id INT NOT NULL,
	BMI DECIMAL NOT NULL,
	PhysicalHealth INT NOT NULL,
	MentalHealth INT NOT NULL,
	AgeCategory VARCHAR NOT NULL,
	Race VARCHAR NOT NULL,
	Diabetic VARCHAR NOT NULL,
	GenHealth VARCHAR NOT NULL,
	SleepTime INT NOT NULL,
	FOREIGN KEY (id) REFERENCES binary_data (id),
	PRIMARY KEY (id)
);
