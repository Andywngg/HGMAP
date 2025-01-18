import React from 'react';
import { Formik, Form } from 'formik';
import * as Yup from 'yup';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import FormField from '../components/FormField';
import Button from '../components/Button';
import toast from 'react-hot-toast';

const LoginSchema = Yup.object().shape({
  email: Yup.string()
    .email('Invalid email')
    .required('Required'),
  password: Yup.string()
    .required('Required'),
});

const Login = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-white">
            Sign in to your account
          </h2>
        </div>

        <Formik
          initialValues={{
            email: '',
            password: '',
          }}
          validationSchema={LoginSchema}
          onSubmit={async (values, { setSubmitting }) => {
            try {
              await login(values.email, values.password);
              toast.success('Login successful!');
              navigate('/dashboard');
            } catch (error) {
              toast.error('Invalid email or password');
            } finally {
              setSubmitting(false);
            }
          }}
        >
          {({ isSubmitting }) => (
            <Form className="mt-8 space-y-6 bg-gray-800 p-8 rounded-lg shadow-xl">
              <div className="space-y-4">
                <FormField
                  name="email"
                  label="Email"
                  type="email"
                  autoComplete="email"
                />
                <FormField
                  name="password"
                  label="Password"
                  type="password"
                  autoComplete="current-password"
                />
              </div>

              <Button
                type="submit"
                isLoading={isSubmitting}
                className="w-full"
              >
                Sign in
              </Button>
            </Form>
          )}
        </Formik>
      </div>
    </div>
  );
};

export default Login; 