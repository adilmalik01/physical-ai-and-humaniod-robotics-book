import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './HomepageFeatures.module.css';
import HeroSection from './HeroSection';
import ModuleCards from './ModuleCards';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI & Embodied Intelligence',
    description: (
      <>
        Learn how to connect AI cognition (LLMs, perception, planning) with robotic embodiment
        (navigation, manipulation, interaction) in real physical environments.
      </>
    ),
  },
  {
    title: 'Applied Learning Approach',
    description: (
      <>
        Each module delivers applied, real-world learning outcomes with hands-on exercises
        rather than theoretical concepts alone.
      </>
    ),
  },
  {
    title: 'Complete System Integration',
    description: (
      <>
        Build toward a final capstone demonstrating a fully autonomous humanoid in simulation
        that responds to voice commands, plans actions, navigates, and manipulates objects.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <>
      <HeroSection />
      <ModuleCards />
      <section className={styles.features}>
        <div className="container">
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}