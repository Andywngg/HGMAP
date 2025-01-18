import React from 'react';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import FormField from '../components/FormField';
import Button from '../components/Button';
import toast from 'react-hot-toast';
import axios from 'axios';
import ImageUpload from '../components/ImageUpload';
import { motion } from 'framer-motion';
import endpoints from '../config/api';
import { FormikHelpers } from 'formik';
import { useNavigate } from 'react-router-dom';

interface ScoutingFormValues {
  teamNumber: string;
  // Auto
  autoScoreCoral: boolean;
  autoScoreAlgae: boolean;
  mustStartSpecificPosition: boolean;
  autoStartingPosition: string;
  // Teleop
  teleopCoralCapability: string;
  teleopAlgaeCapability: string;
  teleopDealgifying: boolean;
  teleopPreference: string;
  teleopCoralLevels: string[];
  // Endgame
  endgameType: string;
  // Robot Specs
  robotWidth: string;
  robotLength: string;
  robotHeight: string;
  robotWeight: string;
  drivetrainType: string;
  // Image
  hasRobotImage: boolean;
  robotImage: File | null;
  scoringPreference: string;
  coralLevels: string[];
  [key: string]: string | boolean | string[] | File | null;
}

const ScoutingSchema = Yup.object().shape({
  teamNumber: Yup.number()
    .required('Team number is required')
    .positive('Must be a positive number')
    .integer('Must be an integer'),
  autoStartingPosition: Yup.string().when('mustStartSpecificPosition', ([value]) => {
    return value ? Yup.string().required('Please specify the starting position') : Yup.string();
  }),
  teleopPreference: Yup.string().required('Required'),
  scoringPreference: Yup.string().when('teleopPreference', ([value]) => {
    return value === 'both' ? Yup.string().required('Required when scoring both') : Yup.string().nullable();
  }),
  coralLevels: Yup.array().of(Yup.string()),
  endgameType: Yup.string().required('Required'),
  robotWidth: Yup.number()
    .required('Width is required')
    .positive('Must be positive'),
  robotLength: Yup.number()
    .required('Length is required')
    .positive('Must be positive'),
  robotHeight: Yup.number()
    .required('Height is required')
    .positive('Must be positive'),
  robotWeight: Yup.number()
    .required('Weight is required')
    .positive('Must be positive'),
  drivetrainType: Yup.string()
    .required('Drivetrain type is required'),
  robotImage: Yup.mixed().when('hasRobotImage', ([value]) => {
    return value ? Yup.mixed().required('Please upload an image') : Yup.mixed();
  }),
});

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const Section: React.FC<{
  title: string;
  children: React.ReactNode;
  delay?: number;
}> = ({ title, children, delay = 0 }) => (
  <motion.div
    initial="hidden"
    animate="visible"
    variants={fadeIn}
    transition={{ duration: 0.5, delay }}
    className="bg-gradient-to-br from-gray-900 to-black p-6 rounded-xl border border-red-900/30 shadow-lg hover:shadow-red-900/20 transition-all duration-300"
  >
    <div className="flex items-center space-x-4 mb-6">
      <div className="h-8 w-1 bg-gradient-to-b from-red-500 to-red-800 rounded-full" />
      <h2 className="text-2xl font-bold bg-gradient-to-r from-red-500 to-red-700 bg-clip-text text-transparent">
        {title}
      </h2>
    </div>
    <div className="space-y-4">
      {children}
    </div>
  </motion.div>
);

const ScoutingForm = () => {
  const navigate = useNavigate();

  const handleSubmit = async (values: ScoutingFormValues, { setSubmitting }: FormikHelpers<ScoutingFormValues>) => {
    try {
      const formData = new FormData();
      
      // Add all form fields to formData
      Object.entries(values).forEach(([key, value]) => {
        if (key === 'robotImage' && value instanceof File) {
          formData.append(key, value);
        } else if (key === 'coralLevels' && Array.isArray(value)) {
          formData.append(key, JSON.stringify(value));
        } else {
          formData.append(key, String(value));
        }
      });

      const response = await axios.post(endpoints.teams.create, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data) {
        toast.success('Team created successfully!');
        navigate('/dashboard');
      }
    } catch (error: any) {
      console.error('Error creating team:', error);
      toast.error(error.response?.data?.error || 'Error creating team');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-black to-black py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-extrabold leading-tight bg-gradient-to-r from-red-500 to-red-700 bg-clip-text text-transparent mb-4">
            Team 1334 Pit Scouting
          </h1>
        </motion.div>

        {/* Main Form */}
        <Formik<ScoutingFormValues>
          initialValues={{
            teamNumber: '',
            autoScoreCoral: false,
            autoScoreAlgae: false,
            mustStartSpecificPosition: false,
            autoStartingPosition: '',
            teleopCoralCapability: '',
            teleopAlgaeCapability: '',
            teleopDealgifying: false,
            teleopPreference: '',
            teleopCoralLevels: [],
            endgameType: '',
            robotWidth: '',
            robotLength: '',
            robotHeight: '',
            robotWeight: '',
            drivetrainType: '',
            hasRobotImage: false,
            robotImage: null,
            scoringPreference: '',
            coralLevels: []
          }}
          validationSchema={ScoutingSchema}
          onSubmit={handleSubmit}
        >
          {({ isSubmitting, values, setFieldValue, errors, touched }) => (
            <Form className="space-y-6">
              {/* Team Number */}
              <Section title="Team Number" delay={0.1}>
                <div className="relative">
                  <FormField
                    name="teamNumber"
                    label=""
                    type="number"
                    placeholder="Enter team number"
                    className="text-3xl font-bold bg-gray-800/50 border-2 border-red-900/30 rounded-lg px-4 py-3 w-full text-white placeholder-gray-500 focus:border-red-500 focus:ring-red-500"
                  />
                  {errors.teamNumber && touched.teamNumber && (
                    <div className="text-red-500 text-sm mt-1">{errors.teamNumber}</div>
                  )}
                </div>
              </Section>

              {/* Auto Section */}
              <Section title="Autonomous" delay={0.2}>
                <div className="grid gap-4">
                  <div className="flex items-center space-x-4 bg-gray-800/50 p-4 rounded-lg border border-red-900/20">
                    <FormField
                      name="autoScoreCoral"
                      label="Can score coral"
                      type="checkbox"
                    />
                    <label className="text-gray-200">Can Score Coral</label>
                  </div>
                  <div className="flex items-center space-x-4 bg-gray-800/50 p-4 rounded-lg border border-red-900/20">
                    <FormField
                      name="autoScoreAlgae"
                      label="Can score algae"
                      type="checkbox"
                    />
                    <label className="text-gray-200">Can Score Algae</label>
                  </div>
                  <div className="flex items-center space-x-4 bg-gray-800/50 p-4 rounded-lg border border-red-900/20">
                    <FormField
                      name="mustStartSpecificPosition"
                      label="Must start in specific position"
                      type="checkbox"
                    />
                    <label className="text-gray-200">Must Start in Specific Position?</label>
                  </div>
                  {values.mustStartSpecificPosition && (
                    <FormField
                      name="autoStartingPosition"
                      label="Starting Position"
                      type="text"
                      placeholder="Enter Starting Position"
                      className="bg-white-800/50 border border-red-900/20"
                    />
                  )}
                </div>
              </Section>

              {/* Teleop Section */}
              <Section title="Teleop" delay={0.3}>
                <div className="space-y-4">
                  <div className="flex items-center space-x-4 bg-gray-800/50 p-4 rounded-lg border border-red-900/20">
                    <FormField
                      name="teleopDealgifying"
                      label=" "
                      type="checkbox"
                    />
                    <label className="text-gray-200">Can Dealgify</label>
                  </div>
                  <FormField
                    name="teleopPreference"
                    label="Scoring Preference"
                    type="radio"
                    options={[
                      { value: 'coral', label: 'Coral Only' },
                      { value: 'algae', label: 'Algae Only' },
                      { value: 'both', label: 'Both' }
                    ]}
                  />
                  
                  <Field name="teleopPreference">
                    {({ field: { value } }: any) => value === 'both' && (
                      <FormField
                        name="scoringPreference"
                        label="Scoring Capability"
                        type="radio"
                        options={[
                          { value: 'betterCoral', label: 'Better at Coral than Algae' },
                          { value: 'betterAlgae', label: 'Better at Algae than Coral' }
                        ]}
                      />
                    )}
                  </Field>

                  <FormField
                    name="coralLevels"
                    label="Coral Scoring Levels"
                    type="checkbox"
                    multiple
                    options={[
                      { value: 'level1', label: 'Level 1' },
                      { value: 'level2', label: 'Level 2' },
                      { value: 'level3', label: 'Level 3' },
                      { value: 'level4', label: 'Level 4' }
                    ]}
                  />
                </div>
              </Section>

              {/* Endgame Section */}
              <Section title="Endgame" delay={0.4}>
                <FormField
                  name="endgameType"
                  label="Endgame Strategy"
                  type="radio"
                  options={[
                    { value: 'deep', label: 'Deep Climb' },
                    { value: 'shallow', label: 'Shallow Climb' },
                    { value: 'none', label: 'No Climb' }
                  ]}
                />
              </Section>

              {/* Robot Specs Section */}
              <Section title="Robot Specifications" delay={0.5}>
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    name="robotWidth"
                    label="Width (inches)"
                    type="number"
                    step="0.1"
                    className="bg-white-800/50 border border-red-900/20"
                  />
                  <FormField
                    name="robotLength"
                    label="Length (inches)"
                    type="number"
                    step="0.1"
                    className="bg-white-800/50 border border-red-900/20"
                  />
                  <FormField
                    name="robotHeight"
                    label="Height (inches)"
                    type="number"
                    step="0.1"
                    className="bg-white-800/50 border border-red-900/20"
                  />
                  <FormField
                    name="robotWeight"
                    label="Weight (lbs)"
                    type="number"
                    step="0.1"
                    className="bg-white-800/50 border border-red-900/20"
                  />
                  <div className="col-span-2">
                    <FormField
                      name="drivetrainType"
                      label="Drivetrain Type"
                      type="select"
                      options={[
                        { value: '', label: 'Select drivetrain type' },
                        { value: 'swerve', label: 'Swerve Drive' },
                        { value: 'tank', label: 'Tank Drive' },
                        { value: 'mecanum', label: 'Mecanum Drive' },
                        { value: 'other', label: 'Other' }
                      ]}
                      className="bg-white-800/50 border border-red-900/20"
                    />
                  </div>
                </div>
              </Section>

              {/* Robot Image Section */}
              <Section title="Robot Image" delay={0.6}>
                <div className="space-y-4">
                  <div className="flex items-center space-x-4 bg-gray-800/50 p-4 rounded-lg border border-red-900/20">
                    <FormField
                      name="hasRobotImage"
                      label="Permission for photo"
                      type="checkbox"
                    />
                    <label className="text-gray-200">Can we take or see a picture of your robot?</label>
                  </div>
                  {values.hasRobotImage && (
                    <div className="mt-4">
                      <ImageUpload
                        name="robotImage"
                        label="Upload Robot Image"
                        onUpload={(file) => setFieldValue('robotImage', file)}
                      />
                    </div>
                  )}
                </div>
              </Section>

              {/* Submit Button */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.7 }}
                className="flex justify-end pt-6"
              >
                <Button
                  type="submit"
                  isLoading={isSubmitting}
                  className="bg-gradient-to-r from-red-600 to-red-800 hover:from-red-700 hover:to-red-900 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-lg transform transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Scouting Data'}
                </Button>
              </motion.div>
            </Form>
          )}
        </Formik>
      </div>
    </div>
  );
};

export default ScoutingForm; 