import React from 'react';
import { Formik, Form } from 'formik';
import * as Yup from 'yup';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import FormField from '../components/FormField';
import Button from '../components/Button';

const RegisterSchema = Yup.object().shape({
  name: Yup.string()
    .min(2, 'Too Short!')
    .max(50, 'Too Long!')
    .required('Required'),
  email: Yup.string()
    .email('Invalid email')
    .required('Required'),
  password: Yup.string()
    .min(8, 'Password must be at least 8 characters')
    .required('Required'),
  teamNumber: Yup.number()
    .required('Required')
    .positive('Must be a positive number')
    .integer('Must be an integer'),
});

const Register = () => {
  const navigate = useNavigate();
  const [error, setError] = React.useState('');

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Create your account
          </h2>
        </div>
        
        {error && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="text-sm text-red-700">{error}</div>
          </div>
        )}

        <Formik
          initialValues={{
            name: '',
            email: '',
            password: '',
            teamNumber: '',
          }}
          validationSchema={RegisterSchema}
          onSubmit={async (values, { setSubmitting }) => {
            try {
              const response = await axios.post('http://localhost:5001/api/auth/register', values);
              console.log('Registration successful:', response.data);
              navigate('/login');
            } catch (error: any) {
              setError(error.response?.data?.error || 'Registration failed');
              console.error('Registration error:', error.response?.data);
            } finally {
              setSubmitting(false);
            }
          }}
        >
          {({ isSubmitting }) => (
            <Form className="mt-8 space-y-6">
              <FormField name="name" label="Name" />
              <FormField name="email" label="Email" type="email" />
              <FormField name="password" label="Password" type="password" />
              <FormField name="teamNumber" label="Team Number" type="number" />
              
              <Button
                type="submit"
                isLoading={isSubmitting}
                className="w-full"
              >
                Register
              </Button>
            </Form>
          )}
        </Formik>
      </div>
    </div>
  );
};

export default Register; 