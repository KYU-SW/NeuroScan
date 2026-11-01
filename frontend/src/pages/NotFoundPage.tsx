import { Link } from "react-router-dom";

import { Layout } from "../components/Layout";
import { ko } from "../i18n/ko";

export function NotFoundPage() {
  return (
    <Layout>
      <section className="panel narrow">
        <h1>{ko.notFound.title}</h1>
        <Link to="/">{ko.notFound.home}</Link>
      </section>
    </Layout>
  );
}
